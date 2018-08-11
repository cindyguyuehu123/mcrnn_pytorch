import os
import torch
import numpy as np
import time


from src.crowd_count import CrowdCounter
from src import network
from src.data_loader import ImageDataLoader
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

data_path =  './data/original/shanghaitech/part_B_final/test_data/images/'
gt_path = './data/original/shanghaitech/part_B_final/test_data/ground_truth_csv/'
model_path = './final_models/mcnn_shtechB_110.h5'

output_dir = './output/'
model_name = os.path.basename(model_path).split('.')[0]
file_results = os.path.join(output_dir,'results_' + model_name + '_.txt')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


net = CrowdCounter()
      
trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
#net.cuda()
net.eval()
mae = 0.0
mse = 0.0

# open the output file
f = open(file_results, 'w') 

#record start time
start = time.time()


#load test data
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=True)

num_of_img = 0
for blob in data_loader:
    num_of_img += 1
    
    # im_data is a matrix of the image, shape 1 x 1 x 768 x 1024 (Shanghaitech dataset group B)
    im_data = blob['data']
    
    # gt_data is the ground truth density map of the training data, shape 192 x 256 (Shanghaitech dataset group B)
    gt_data = blob['gt_density']
    
    img_name = blob['fname']
    
    # print the name of image filename
    print(img_name)
    f.write(img_name + "\n")
    
    density_map = net(im_data, gt_data)
    density_map = density_map.data.cpu().numpy()
    gt_count = np.sum(gt_data)
    et_count = np.sum(density_map)
    
    print("Ground truth count: ", gt_count)
    print("Predicted    count: ", et_count)
    print("diff              : ", abs(gt_count - et_count))
    print("diff/ground truth : ", "%.2f" %( abs(gt_count - et_count) / gt_count), "\n")
    
    f.write("Ground truth count: " + str(gt_count) + "\n")
    f.write("Predicted    count: " +  str(et_count) + "\n")
    f.write("diff              : " +  str(abs(gt_count - et_count)) + "\n")
    f.write("diff/ground truth : " + "%.2f" %( abs(gt_count - et_count) / gt_count) + "\n \n")
    
    
    mae += abs(gt_count-et_count)
    mse += ((gt_count-et_count)*(gt_count-et_count))
    if vis:
        utils.display_results(im_data, gt_data, density_map)
    if save_output:
        #utils.save_density_map(density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
        #utils.save_results(im_data, gt_data, density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')
        utils.save_blend_results(im_data, gt_data, density_map, output_dir, 'output_' + blob['fname'].split('.')[0] + '.png', 0.7)
        
#record end time
end = time.time()
running_time = end-start

#print total time, average time, and fps

print("\n", "Shanghaitech_B", "\n")

print("Total running time:", "%.2f" %running_time, "s")
print("Average running time:", "%.2f" %(running_time/num_of_img), "s")
print("FPS = ", "%.2f" %(num_of_img/running_time))

mae = mae/data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples())
print('\nMAE: %0.2f, MSE: %0.2f' % (mae,mse))




f.write('MAE: %0.2f, MSE: %0.2f' % (mae,mse))
f.close()