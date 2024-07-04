from torchvision import transforms

def get_eval_transforms(mean, std, target_img_size = -1):
	trsforms = []
	color = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
	random_color = transforms.RandomApply([color], p = 0.7)
	if target_img_size > 0:
		trsforms.append(transforms.Resize(target_img_size))
	trsforms.append(transforms.ToTensor())
	trsforms.append(transforms.Normalize(mean, std))
	trsforms.append(random_color)
	trsforms = transforms.Compose(trsforms)
	

	return trsforms