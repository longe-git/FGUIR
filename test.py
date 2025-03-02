import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
# from ptflops import get_model_complexity_info
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
# from models import *
# from models.FGUIR import Net
from models.Net5 import Net5

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='Net5', type=str, help='model name')
parser.add_argument('--weights', default='wo-cl2-psnr2470.pth', type=str, help='model pretrained weights')
parser.add_argument('--num_workers', default=1, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./save_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./out/', type=str, help='path to results saving') # './result/'
parser.add_argument('--patch_size', default=[256, 256], type=list, help='image patch size')
parser.add_argument('--dataset', default='./val/', type=str, help='dataset name')
args = parser.parse_args()


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict


def test(test_loader, network, result_dir):  # result_dir is 'result/数据集/模型名称'
	PSNR = AverageMeter()
	SSIM = AverageMeter()
	torch.cuda.empty_cache()
	network.eval()

	os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)  # 创建存图片的文件夹
	f_result = open(os.path.join(result_dir, 'results.csv'), 'w')  # 创建存数据结果的文件夹

	num_batch = len(test_loader)
	for idx, batch in tqdm(enumerate(test_loader), total=num_batch):
		input = batch['input'].cuda()
		gt = batch['gt'].cuda()
		filename = batch['filename'][0]

		with torch.no_grad():
			output = network(input).clamp_(-1, 1)

			output = output * 0.5 + 0.5
			target = gt * 0.5 + 0.5

			psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()

			_, _, H, W = output.size()
			down_ratio = max(1, round(min(H, W) / 256))
			ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
							F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
							data_range=1, size_average=False).item()

		PSNR.update(psnr_val)
		SSIM.update(ssim_val)

	# 	f_result.write('%s,%.02f,%.03f\n'%(filename, psnr_val, ssim_val))
	# 	out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
	# 	write_img(os.path.join(result_dir, 'imgs', filename), out_img)
	#
	# f_result.close()

	# os.rename(os.path.join(result_dir, 'results.csv'),
	# 		  os.path.join(result_dir, '%.02f | %.04f.csv'%(PSNR.avg, SSIM.avg)))

	return PSNR.avg, SSIM.avg


if __name__ == '__main__':
	network = eval(args.model.replace('-', '_'))()
	network.cuda()

	# macs, params = get_model_complexity_info(network, (3, 224, 224), as_strings=True,
	# 										 print_per_layer_stat=True, verbose=True)
	# print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
	# print('{:<30}  {:<8}'.format('Number of parameters: ', params))

	dataset_dir = './data/'
	test_dataset = PairLoader(dataset_dir, args.dataset, 'valid', args.patch_size)
	test_loader = DataLoader(test_dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)
	best_psnr, best_ssim, best_pt = 0, 0, ''
	print('测试模型为', args.model)
	# pth_list = os.listdir(os.path.join(args.save_dir, args.model))
	i = 0

	# 单独测试的方法
	# pth_path = os.path.join(args.save_dir, args.model, args.weights)
	pth_path = os.path.join(args.save_dir, 'Net5', args.weights)
	network.load_state_dict(single(pth_path))
	result_dir = os.path.join(args.result_dir, args.dataset, 'Net5')
	cur_psnr, cur_ssim = test(test_loader, network, result_dir)
	cur_psnr = float('{:.02f}'.format(cur_psnr))
	cur_ssim = float('{:.03f}'.format(cur_ssim))
	print('测试结果psnr{}, ssim{}'.format(cur_psnr, cur_ssim))

	# 批量测试
	# for pth in pth_list:
	# 	print('当前测试{}，测试进度{}/{}'.format(pth, i, len(pth_list)))
	# 	pth_path = os.path.join(args.save_dir, args.model, pth)
	#
	# 	if os.path.exists(pth_path):
	# 		network.load_state_dict(single(pth_path))
	# 	else:
	# 		print('==> 不存在权重信息!')
	# 		exit(0)
	# 	result_dir = os.path.join(args.result_dir, args.dataset, args.model)
	# 	# if not os.path.exists(os.path.join(args.result_dir, args.dataset)):
	# 	# 	os.mkdir(os.path.join(args.result_dir, args.dataset))
	# 	if not os.path.exists(result_dir):
	# 		os.makedirs(result_dir)
	# 	cur_psnr, cur_ssim = test(test_loader, network, result_dir)
	# 	cur_psnr = float('{:.02f}'.format(cur_psnr))
	# 	cur_ssim = float('{:.03f}'.format(cur_ssim))
	# 	# print('PSNR: {cur_psnr:.02f}\t''SSIM: {cur_ssim:.03f})'.format(cur_psnr, cur_ssim))
	#
	# 	if cur_psnr > best_psnr or (cur_psnr == best_psnr and cur_ssim > best_ssim):
	# 		new_pth = args.model+pth
	# 		# os.rename(pth_path, os.path.join(args.save_dir, args.model, new_pth))
	# 		best_psnr, best_ssim = cur_psnr, cur_ssim
	# 		best_pt = pth
	# 	print('测试结果psnr{}, ssim{}，最好的指标是{}的psnr{}, ssim{}'.format(cur_psnr, cur_ssim, best_pt, best_psnr, best_ssim))
	# 	i += 1
