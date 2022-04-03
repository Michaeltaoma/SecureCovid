from model import pretrained, covid_net, cnn


def load_model(device, model_name, class_names):
	if model_name.__eq__("dense"):
		shadow = pretrained.dense_shadow(device, class_names, pretrained=True)
	elif model_name.__eq__("covidnet"):
		shadow = covid_net.CovidNet(model='small', n_classes=2)
		shadow = shadow.to(device)
	else:
		shadow = cnn.ConvNet()

	return shadow
