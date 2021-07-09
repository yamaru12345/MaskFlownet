import os
import sys

import argparse

from timeit import default_timer
import yaml
import hashlib
import socket

# ======== PLEASE MODIFY ========
# where is the repo
repoRoot = r'.'
# to CUDA\vX.Y\bin
#os.environ['PATH'] = r'path\to\your\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin' + ';' + os.environ['PATH']
# Flying Chairs Dataset
#chairs_path = r'path\to\your\FlyingChairs_release\data'
#chairs_split_file = r'path\to\your\FlyingChairs_release\FlyingChairs_train_val.txt'

import numpy as np
import mxnet as mx

# data readers
from reader.chairs import binary_reader, trainval, ppm, flo
from reader import sintel, kitti, hd1k, things3d, orig_loader
import cv2

model_parser = argparse.ArgumentParser(add_help=False)
training_parser = argparse.ArgumentParser(add_help=False)
training_parser.add_argument('--batch', type=int, default=8, help='minibatch size of samples per device')

parser = argparse.ArgumentParser(parents=[model_parser, training_parser])

parser.add_argument('config', type=str, nargs='?', default=None)
parser.add_argument('--dataset_cfg', type=str, default='chairs.yaml')
# proportion of data to be loaded
# for example, if shard = 4, then one fourth of data is loaded
# ONLY for things3d dataset (as it is very large)
parser.add_argument('-s', '--shard', type=int, default=1, help='')

parser.add_argument('-g', '--gpu_device', type=str, default='', help='Specify gpu device(s)')
parser.add_argument('-c', '--checkpoint', type=str, default=None, 
	help='model checkpoint to load; by default, the latest one.'
	'You can use checkpoint:steps to load to a specific steps')
parser.add_argument('--clear_steps', action='store_true')
# the choice of network
parser.add_argument('-n', '--network', type=str, default='MaskFlownet')
parser.add_argument('--samples', type=int, default=-1)
# three modes
parser.add_argument('--debug', action='store_true', help='Do debug')
parser.add_argument('--valid', action='store_true', help='Do validation')
parser.add_argument('--predict', action='store_true', help='Do prediction')
# inference resize for validation and prediction
parser.add_argument('--resize', type=str, default='')

args = parser.parse_args()
ctx = [mx.cpu()] if args.gpu_device == '' else [mx.gpu(gpu_id) for gpu_id in map(int, args.gpu_device.split(','))]
infer_resize = [int(s) for s in args.resize.split(',')] if args.resize else None

import network.config
# load network configuration
print(f"config: {os.path.join(repoRoot, 'network', 'config', args.config)}")
with open(os.path.join(repoRoot, 'network', 'config', args.config)) as f:
	config =  network.config.Reader(yaml.load(f))
# load training configuration
with open(os.path.join(repoRoot, 'network', 'config', args.dataset_cfg)) as f:
	dataset_cfg = network.config.Reader(yaml.load(f))

validation_steps = dataset_cfg.validation_steps.value
checkpoint_steps = dataset_cfg.checkpoint_steps.value

# create directories
def mkdir(path):
	if not os.path.exists(path):
		os.makedirs(path)
mkdir('logs')
mkdir(os.path.join('logs', 'val'))
mkdir(os.path.join('logs', 'debug'))
mkdir('weights')
mkdir('flows')

# find checkpoint
import path
import logger
steps = 0
if args.checkpoint is not None:
	if ':' in args.checkpoint:
		prefix, steps = args.checkpoint.split(':')
	else:
		prefix = args.checkpoint
		steps = None
	log_file, run_id = path.find_log(prefix)	
	if steps is None:
		checkpoint, steps = path.find_checkpoints(run_id)[-1]
	else:
		checkpoints = path.find_checkpoints(run_id)
		try:
			checkpoint, steps = next(filter(lambda t : t[1] == steps, checkpoints))
		except StopIteration:
			print('The steps not found in checkpoints', steps, checkpoints)
			sys.stdout.flush()
			raise StopIteration
	steps = int(steps)
	if args.clear_steps:
		steps = 0
	else:
		_, exp_info = path.read_log(log_file)
		exp_info = exp_info[-1]
		for k in args.__dict__:
			if k in exp_info and k in ('tag',):
				setattr(args, k, eval(exp_info[k]))
				print('{}={}, '.format(k, exp_info[k]), end='')
		print()
	sys.stdout.flush()
# generate id
if args.checkpoint is None or args.clear_steps:
	uid = (socket.gethostname() + logger.FileLog._localtime().strftime('%b%d-%H%M') + args.gpu_device)
	tag = hashlib.sha224(uid.encode()).hexdigest()[:3] 
	run_id = tag + logger.FileLog._localtime().strftime('%b%d-%H%M')

# initiate
from network import get_pipeline
print(f'config.optimizer: {config.optimizer.learning_rate.value}')
pipe = get_pipeline(args.network, ctx=ctx, config=config)
lr_schedule = dataset_cfg.optimizer.learning_rate.get(None)
if lr_schedule is not None:
	pipe.lr_schedule = lr_schedule

print(f'lr_schedule: {lr_schedule}, {pipe.lr_schedule}')

# load parameters from given checkpoint
if args.checkpoint is not None:
	print('Load Checkpoint {}'.format(checkpoint))
	sys.stdout.flush()
	network_class = getattr(config.network, 'class').get()
	# if train the head-stack network for the first time
	if network_class == 'MaskFlownet' and args.clear_steps and dataset_cfg.dataset.value == 'chairs':
		print('load the weight for the head network only')
		pipe.load_head(checkpoint)
	else:
		print('load the weight for the network')
		pipe.load(checkpoint)
	if network_class == 'MaskFlownet':
		print('fix the weight for the head network')
		pipe.fix_head()
	sys.stdout.flush()
	if not args.valid and not args.predict and not args.clear_steps:
		pipe.trainer.step(100, ignore_stale_grad=True)
		pipe.trainer.load_states(checkpoint.replace('params', 'states'))


# ======== If to do prediction ========

if args.predict:
	import predict
	checkpoint_name = os.path.basename(checkpoint).replace('.params', '')
	predict.predict(pipe, os.path.join(repoRoot, 'flows', checkpoint_name), batch_size=args.batch, resize = infer_resize)
	sys.exit(0)


# ======== If to do validation ========

def validate():
	validation_result = {}
	for dataset_name in validation_datasets:
		validation_result[dataset_name] = pipe.validate(*validation_datasets[dataset_name], batch_size = args.batch)
	return validation_result

if args.valid:
	log = logger.FileLog(os.path.join(repoRoot, 'logs', 'val', '{}.val.log'.format(run_id)), screen=True)
	
	# sintel
	sintel_dataset = sintel.list_data()
	for div in ('training2', 'training'):
		for k, dataset in sintel_dataset[div].items():
			img1, img2, flow, mask = [[sintel.load(p) for p in data] for data in zip(*dataset)]
			val_epe = pipe.validate(img1, img2, flow, mask, batch_size=args.batch, resize = infer_resize)
			log.log('steps={}, sintel.{}.{}:epe={}'.format(steps, div, k, val_epe))
			sys.stdout.flush()
	
	# kitti
	read_resize = (370, 1224) # if infer_resize is None else infer_resize
	for kitti_version in ('2012', '2015'):
		dataset = kitti.read_dataset(editions = kitti_version, parts = 'mixed', resize = read_resize)
		val_epe = pipe.validate(dataset['image_0'], dataset['image_1'], dataset['flow'], dataset['occ'], batch_size=args.batch, resize = infer_resize, return_type = 'epe')	
		log.log('steps={}, kitti.{}:epe={}'.format(steps, kitti_version, val_epe))
		sys.stdout.flush()
		val_epe = pipe.validate(dataset['image_0'], dataset['image_1'], dataset['flow'], dataset['occ'], batch_size=args.batch, resize = infer_resize, return_type = 'kitti')	
		log.log('steps={}, kitti.{}:kitti={}'.format(steps, kitti_version, val_epe))
		sys.stdout.flush()

	log.close()
	sys.exit(0)


# ======== If to do training ========

# load training/validation datasets
validation_datasets = {}
samples = 32 if args.debug else -1
t0 = default_timer()

if dataset_cfg.dataset.value == 'sintel':
	batch_size = 4
	print('loading sintel dataset ...')
	sys.stdout.flush()

	orig_shape = [436, 1024]
	num_kitti = dataset_cfg.kitti.get(0)
	num_hd1k = dataset_cfg.hd1k.get(0)
	subsets = ('training' if dataset_cfg.train_all.get(False) else 'training1', 'training2')

	# training
	trainImg1 = []
	trainImg2 = []
	trainFlow = []
	trainMask = []
	sintel_dataset = sintel.list_data()
	for k, dataset in sintel_dataset[subsets[0]].items():
		dataset = dataset[:samples]
		img1, img2, flow, mask = [[sintel.load(p) for p in data] for data in zip(*dataset)]
		trainImg1.extend(img1)
		trainImg2.extend(img2)
		trainFlow.extend(flow)
		trainMask.extend(mask)
	trainSize = len(trainMask)
	training_datasets = [(trainImg1, trainImg2, trainFlow, trainMask)] * (batch_size - num_kitti - num_hd1k)

	resize_shape = (1024, dataset_cfg.resize_shape.get(436))
	if num_kitti > 0:
		print('loading kitti dataset ...')
		sys.stdout.flush()
		editions = '2015'
		dataset = kitti.read_dataset(resize = resize_shape, samples = samples, editions = editions)
		trainSize += len(dataset['flow'])
		training_datasets += [(dataset['image_0'], dataset['image_1'], dataset['flow'], dataset['occ'])] * num_kitti

	if num_hd1k > 0:
		print('loading hd1k dataset ...')
		sys.stdout.flush()
		dataset = hd1k.read_dataset(resize = resize_shape, samples = samples)
		trainSize += len(dataset['flow'])
		training_datasets += [(dataset['image_0'], dataset['image_1'], dataset['flow'], dataset['occ'])] * num_hd1k

	# validation
	validationSize = 0
	for k, dataset in sintel_dataset[subsets[1]].items():
		dataset = dataset[:samples]
		img1, img2, flow, mask = [[sintel.load(p) for p in data] for data in zip(*dataset)]
		validationSize += len(flow)
		validation_datasets['sintel.' + k] = (img1, img2, flow, mask)

elif dataset_cfg.dataset.value == 'orig':
	batch_size = 4
	print('loading orig dataset ...')
	sys.stdout.flush()

	orig_shape = [704, 2624]

	# training
	trainImg1 = []
	trainImg2 = []
	trainFlow = []
	trainMask = []
	orig_dataset = orig_loader.list_data()
	orig_dataset = [orig_dataset[0][:args.samples], orig_dataset[1][:args.samples]]
	print(len(orig_dataset[0]))

	train_data_count = 0
	for dataset in orig_dataset[0]:

		trainImg1.append(orig_loader.load(dataset[0]))
		trainImg2.append(orig_loader.load(dataset[1]))
		trainFlow.append(orig_loader.load(dataset[2]))
		trainMask.append(orig_loader.load(dataset[3]))
		train_data_count += 1

	trainSize = len(trainMask)
	training_datasets = [(trainImg1, trainImg2, trainFlow, trainMask)] * batch_size

	# validation
	validationSize = 0

	valImg1 = []
	valImg2 = []
	valFlow = []
	valMask = []

	for dataset in orig_dataset[1]:
		valImg1.append(orig_loader.load(dataset[0]))
		valImg2.append(orig_loader.load(dataset[1]))
		valFlow.append(orig_loader.load(dataset[2]))
		valMask.append(orig_loader.load(dataset[3]))
		validationSize += 1

	validation_datasets['orig.0'] = (valImg1, valImg2, valFlow, valMask)

else:
	raise NotImplementedError

print('Using {}s'.format(default_timer() - t0))
sys.stdout.flush()

# 
assert batch_size % len(ctx) == 0
batch_size_card = batch_size // len(ctx)

orig_shape = dataset_cfg.orig_shape.get(orig_shape)
target_shape = dataset_cfg.target_shape.get([shape_axis + (64 - shape_axis) % 64 for shape_axis in orig_shape])
print('original shape: ' + str(orig_shape))
print('target shape: ' + str(target_shape))
sys.stdout.flush()

# create log file
log = logger.FileLog(os.path.join(repoRoot, 'logs', 'debug' if args.debug else '', '{}.log'.format(run_id)))
log.log('start={}, train={}, val={}, host={}, batch={}'.format(steps, trainSize, validationSize, socket.gethostname(), batch_size))
information = ', '.join(['{}={}'.format(k, repr(args.__dict__[k])) for k in args.__dict__])
log.log(information)

# implement data augmentation
import augmentation

# chromatic augmentation
aug_func = augmentation.ColorAugmentation
if dataset_cfg.dataset.value == 'sintel':
	color_aug = aug_func(contrast_range=(-0.4, 0.8), brightness_sigma=0.1, channel_range=(0.8, 1.4), batch_size=batch_size_card, 
		shape=target_shape, noise_range=(0, 0), saturation=0.5, hue=0.5, eigen_aug = False)
elif dataset_cfg.dataset.value == 'kitti':
	color_aug = aug_func(contrast_range=(-0.2, 0.4), brightness_sigma=0.05, channel_range=(0.9, 1.2), batch_size=batch_size_card, 
		shape=target_shape, noise_range=(0, 0.02), saturation=0.25, hue=0.1, gamma_range=(-0.5, 0.5), eigen_aug = False)
else:
	color_aug = aug_func(contrast_range=(-0.4, 0.8), brightness_sigma=0.1, channel_range=(0.8, 1.4), batch_size=batch_size_card, 
		shape=target_shape, noise_range=(0, 0.04), saturation=0.5, hue=0.5, eigen_aug = False)
color_aug.hybridize()

# geometric augmentation
aug_func = augmentation.GeometryAugmentation
if dataset_cfg.dataset.value == 'sintel':
	geo_aug = aug_func(angle_range=(-17, 17), zoom_range=(1 / 1.5, 1 / 0.9), aspect_range=(0.9, 1 / 0.9), translation_range=0.1,
											target_shape=target_shape, orig_shape=orig_shape, batch_size=batch_size_card,
											relative_angle=0.25, relative_scale=(0.96, 1 / 0.96), relative_translation=0.25
											)
elif dataset_cfg.dataset.value == 'kitti':
	geo_aug = aug_func(angle_range=(-5, 5), zoom_range=(1 / 1.25, 1 / 0.95), aspect_range=(0.95, 1 / 0.95), translation_range=0.05,
											target_shape=target_shape, orig_shape=orig_shape, batch_size=batch_size_card,
											relative_angle=0.25, relative_scale=(0.98, 1 / 0.98), relative_translation=0.25
											)
else:
	geo_aug = aug_func(angle_range=(-17, 17), zoom_range=(0.5, 1 / 0.9), aspect_range=(0.9, 1 / 0.9), translation_range=0.1,
											target_shape=target_shape, orig_shape=orig_shape, batch_size=batch_size_card,
											relative_angle=0.25, relative_scale=(0.96, 1 / 0.96), relative_translation=0.25
											)
geo_aug.hybridize()

def index_generator(n):
	indices = np.arange(0, n, dtype=np.int)
	while True:
		np.random.shuffle(indices)
		yield from indices

class MovingAverage:
	def __init__(self, ratio=0.95):
		self.sum = 0
		self.weight = 1e-8
		self.ratio = ratio

	def update(self, v):
		self.sum = self.sum * self.ratio + v
		self.weight = self.weight * self.ratio + 1

	@property
	def average(self):
		return self.sum / self.weight

class DictMovingAverage:
	def __init__(self, ratio=0.95):
		self.sum = {}
		self.weight = {}
		self.ratio = ratio

	def update(self, v):
		for key in v:
			if key not in self.sum:
				self.sum[key] = 0
				self.weight[key] = 1e-8
			self.sum[key] = self.sum[key] * self.ratio + v[key]
			self.weight[key] = self.weight[key] * self.ratio + 1

	@property
	def average(self):
		return dict([(key, self.sum[key] / self.weight[key]) for key in self.sum])
	
loading_time = MovingAverage()
total_time = MovingAverage()
train_avg = DictMovingAverage()

from threading import Thread
from queue import Queue

def iterate_data(iq, dataset):
	gen = index_generator(len(dataset[0]))
	while True:
		i = next(gen)
		data = [item[i] for item in dataset]
		space_x, space_y = data[0].shape[0] - orig_shape[0], data[0].shape[1] - orig_shape[1]
		crop_x, crop_y = space_x and np.random.randint(space_x), space_y and np.random.randint(space_y)
		data = [np.transpose(arr[crop_x: crop_x + orig_shape[0], crop_y: crop_y + orig_shape[1]], (2, 0, 1)) for arr in data]
		# vertical flip
		if np.random.randint(2):
			data = [arr[:, :, ::-1] for arr in data]
			data[2] = np.stack([-data[2][0, :, :], data[2][1, :, :]], axis = 0)
		iq.put(data)

def batch_samples(iqs, oq, batch_size):
	while True:
		data_batch = []
		for iq in iqs:
			for i in range(batch_size // len(iqs)):
				data_batch.append(iq.get())
		oq.put([np.stack(x, axis=0) for x in zip(*data_batch)])

def remove_file(iq):
	while True:
		f = iq.get()
		try:
			os.remove(f)
		except OSError as e:
			log.log('Remove failed' + e)

batch_queue = Queue(maxsize=10)
remove_queue = Queue(maxsize=50)

def start_daemon(thread):
	thread.daemon = True
	thread.start()

data_queues = [Queue(maxsize=100) for _ in training_datasets]
for data_queue, dataset in zip(data_queues, training_datasets):
	start_daemon(Thread(target=iterate_data, args=(data_queue, dataset)))

start_daemon(Thread(target=remove_file, args=(remove_queue,)))
for i in range(1):
	start_daemon(Thread(target=batch_samples, args=(data_queues, batch_queue, batch_size)))

t1 = None
checkpoints = []
while True:
	steps += 1
	
	if not pipe.set_learning_rate(steps):
		sys.exit(0)
	batch = []
	t0 = default_timer()
	if t1:
		total_time.update(t0 - t1)
	t1 = t0
	batch = batch_queue.get()
	loading_time.update(default_timer() - t0)

	# # validationのベースライン取得用--------------------------------#
	# val_result = validate()
	# print(val_result.items())
	# input()


	# with or without the given invalid mask
	if len(batch) == 4:
		img1, img2, flow, mask = [batch[i] for i in range(4)]
		train_log = pipe.train_batch(img1, img2, flow, geo_aug, color_aug, mask = mask)
	else:
		img1, img2, flow = [batch[i] for i in range(3)]
		train_log = pipe.train_batch(img1, img2, flow, geo_aug, color_aug)

	# update log
	if steps <= 20 or steps % 50 == 0:
		train_avg.update(train_log)
		log.log('steps={}{}, total_time={:.2f}'.format(steps, ''.join([', {}={}'.format(k, v) for k, v in train_avg.average.items()]), total_time.average))

	# do valiation
	if steps % validation_steps == 0 or steps <= 1:
		val_result = None
		if validationSize > 0:
			val_result = validate()
			log.log('steps={}{}'.format(steps, ''.join([', {}={}'.format(k, v) for k, v in val_result.items()])))

		# save parameters
		if steps % checkpoint_steps == 0:
			prefix = os.path.join(repoRoot, 'weights', '{}_{}'.format(run_id, steps))
			pipe.save(prefix)
			checkpoints.append(prefix)

			# # remove the older checkpoints
			# while len(checkpoints) > 3:
			# 	prefix = checkpoints[0]
			# 	checkpoints = checkpoints[1:]
			# 	remove_queue.put(prefix + '.params')
			# 	remove_queue.put(prefix + '.states')
