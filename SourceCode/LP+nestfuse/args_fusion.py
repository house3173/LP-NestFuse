
class args():
	# training args
	epochs = 2  #"number of training epochs, default is 2"
	batch_size = 1  #"batch size for training, default is 4"
	# dataset = "/data/Disk_B/MSCOCO2014/train2014/"  # the dataset path in your computer
	dataset = r'F:\database\MS-COCO2014\train2014'
	HEIGHT = 256
	WIDTH = 256

	save_model_dir_autoencoder = "models/nestfuse_autoencoder"
	save_loss_dir = './models/loss_autoencoder/'

	cuda = 1
	ssim_weight = [1,10,100,1000,10000]
	ssim_path = ['1e0', '1e1', '1e2', '1e3', '1e4']

	lr = 1e-4  #"learning rate, default is 0.001"
	lr_light = 1e-4  # "learning rate, default is 0.001"
	log_interval = 10  #"number of images after which the training loss is logged, default is 500"
	resume = None

	# for test, model_default is the model used in paper
	# model_default = 'C:/PVH/SourceCode/LP+nestfuse/models/nest_fused_2888.model'
	model_default = 'C:/PVH/SourceCode/LP+nestfuse/models/nestfuse_1e2.model'
	# model_default = 'C:/PVH/SourceCode/LP+nestfuse/models/en_dense_de_dense_1epoch.model'
	# model_default = 'C:/PVH/SourceCode/LP+nestfuse/models/ssim_100.model'
	# model_default = 'C:/PVH/SourceCode/LP+nestfuse/models/loss_with_algebra_001.model'
	# model_default = 'C:/PVH/SourceCode/LP+nestfuse/models/loss_with_mssim_algebra_wight_1.model'
	# model_default = 'C:/PVH/SourceCode/LP+nestfuse/models/msrs_image_aug.model'
	# model_default = 'C:/PVH/SourceCode/LP+nestfuse/models/train_in_image_augment.model'

	model_deepsuper = './models/nestfuse_1e2_deep_super.model'


