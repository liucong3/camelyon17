
def load_model():
	import torch
	model_path = 'checkpoint/ckpt.t7'
	print('eval: loading model from: %s' % model_path)
	checkpoint = torch.load(model_path)
	net = checkpoint['net']
	threshold = checkpoint['threshold']
	print('threshold=%f' % threshold)
	net.eval()
	return net, threshold


def load_slide_and_mask(slide_path, patch_size, mask_level):
	import openslide, cv2, numpy as np
	print('eval: loading slide: %s' % slide_path)	
	slide = openslide.OpenSlide(slide_path)
	# tissue_mask
	slide_lv = slide.read_region((0, 0), mask_level, slide.level_dimensions[mask_level])
	slide_lv = cv2.cvtColor(np.array(slide_lv), cv2.COLOR_RGBA2RGB)
	slide_lv = cv2.cvtColor(slide_lv, cv2.COLOR_BGR2HSV)
	slide_lv = slide_lv[:, :, 1]
	_, tissue_mask = cv2.threshold(slide_lv, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return slide, tissue_mask
	
def make_heatmap(tumor_location, result_file, width, height):
	import cv2, numpy as np
	heatmap = np.zeros([height, width])
	for w, h in tumor_location:
		heatmap[h, w] = 255
	cv2.imwrite(result_file, heatmap)

def eval(slide_file, result_file, update_progress):
	patch_size = 304
	mask_level = 4

	net, threshold = load_model()
	slide, tissue_mask = load_slide_and_mask(slide_file, patch_size, mask_level)

	import torch, torchvision, torchvision.transforms as transforms, numpy as np
	from torch.autograd import Variable 

	trans_test = transforms.Compose([transforms.ToTensor(),])
	width, height = np.array(slide.level_dimensions[0]) // patch_size
	step = int(patch_size / (2 ** mask_level))
	margin = 3
	normal_threshold = 0.1
	tissue_location = []
	for i in range(width):
		for j in range(height):
			if i < margin or j < margin or i >= width - margin or j >= height - margin: continue
			tissue_mask_sum = tissue_mask[step * j : step * (j+1), step * i : step * (i+1)].sum()
			mask_max = step * step * 255
			tissue_mask_ratio = tissue_mask_sum / mask_max

			# extract normal patch
			if tissue_mask_ratio > normal_threshold:
				tissue_location.append((i, j))
	tumor_location = []
	for t in range(len(tissue_location)):
		update_progress(100 * (t + 1) / len(tissue_location))
		(i, j) = tissue_location[t]
		patch = slide.read_region((patch_size*i, patch_size*j), 0, (patch_size, patch_size))
		# patch.save('temp.png')
		# patch = cv2.imread('temp.png', cv2.IMREAD_COLOR)
		patch = np.asarray(patch)
		inputs = trans_test(patch)
		inputs = inputs[:-1]
		# print(inputs.size())
		inputs.unsqueeze_(0)
		if torch.cuda.is_available():
			inputs = inputs.cuda()
		inputs = Variable(inputs, volatile=True)
		outputs = net(inputs)
		outputs = outputs.cpu().squeeze()
		outputs += (1 - threshold)
		# print('output=%f' % outputs)
		outputs = torch.floor(outputs)
		assert outputs.numel() == 1
		outputs = outputs.data.view(-1)[0]
		if outputs == 1:
			tumor_location.append((i, j))
	make_heatmap(tumor_location, result_file, width, height)
	return width, height

if __name__ == '__main__':
	slide_file = '../../training/tif/patient_010_node_0.tif'
	result_file = 'result.png'
	eval(slide_file, result_file, lambda x: print('\r%d%%' % int(x), end=''))

