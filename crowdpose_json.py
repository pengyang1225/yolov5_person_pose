import cv2
# % matplotlib
# inline
import matplotlib.pyplot as plt
import json

json_data = open('/home/py/code/github/yolov5-pose/crowdpose/json/crowdpose_train.json', 'r')
json_data = json.load(json_data)

print(json_data.keys())
# dict_keys(['images', 'annotations', 'categories'])

images_list = json_data['images']
annotations_list = json_data['annotations']
categories_list = json_data['categories']

print('images lenth:', len(images_list))
print('annotations lenth:', len(annotations_list))
print('categories lenth:', len(categories_list))

# images lenth: 2000
# annotations lenth: 8527
# categories lenth: 1

print('categories:', categories_list)

# categories: [{'supercategory': 'person', 'id': 1, 'name': 'person', 'keypoints': ####['left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', #'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', #'right_ankle', 'head', 'neck'], 'skeleton': [[16, 14], [14, 12], [17, 15], [15, 13], [12, #13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11]]}]

print(images_list[0].keys())
print(annotations_list[0].keys())

# dict_keys(['file_name', 'id', 'height', 'width', 'crowdIndex'])
# dict_keys(['num_keypoints', 'iscrowd', 'keypoints', 'image_id', 'bbox', 'category_id', 'id'])

vis_id = 100597
save_txt_dir='/home/py/code/github/yolov5-pose/crowdpose/train/train_txt'

def find_filename(img_id, meta):
    for block in meta:
        # print(block)
        if block['id'] == img_id:
            return block['file_name'], block['crowdIndex']
        continue
    return None, None


def get_annokpts(img_id, meta):
    kpts = []
    bboxes = []
    for block in meta:
        if block['image_id'] == img_id:
            kpts.append(block['keypoints'])
            bboxes.append(block['bbox'])
        continue
    return kpts, bboxes


def vis_box(img, bboxes):
    for box in bboxes:
        x0, y0, x1, y1 = box
        cv2.rectangle(img, (int(x0), int(y0)), (int(x1+x0), int(y1+y0)), (0, 255, 0), thickness=2, lineType=cv2.LINE_AA)  # 12
       # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness=tl, lineType=cv2.LINE_AA)
    return img


def vis_keypoints(img, kpts, crowdIndex):
    links = [[0, 2], [1, 3], [2, 4], [3, 5], [6, 8], [8, 10], [7, 9], [9, 11], [12, 13], [0, 13], [1, 13], [6, 13],
             [7, 13]]
    for kpt in kpts:
        x_ = kpt[0::3]
        y_ = kpt[1::3]
        v_ = kpt[2::3]
        for order1, order2 in links:
            if v_[order1] > 0 and v_[order2] > 0:
                img = img = cv2.line(img, (x_[order1], y_[order1]), (x_[order2], y_[order2]), color=[100, 255, 255],
                                     thickness=2, lineType=cv2.LINE_AA)
        aaa=0
        for x, y, v in zip(x_, y_, v_):
            if int(v) > 0:
                cv2.putText(img,str(aaa), (int(x), int(y)),cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 255), 1)
                img = cv2.circle(
                    img, (int(x), int(y)),
                    radius=3, color=[255, 0, 255], thickness=-1, lineType=cv2.LINE_AA)
            aaa +=1

    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, 'crowdIndex: ' + str(crowdIndex), (0, 50), font, 1, (0, 0, 0),1)

    return img

def json2txt(kpts,bboxes,txt_name,w,h):
    with open(txt_name, "w") as f:
        for i ,box in enumerate(bboxes):
            cen_x =(box[0] +box[2]/2)/w
            cen_y =(box[1] +box[3]/2)/h
            box_w =box[2]/w
            box_h =box[3]/h
            str_box = '0' +" " +str(cen_x) +' ' +str(cen_y) +' ' +str(box_w) +' ' +str(box_h)
            kpt =kpts[i]
            x_ = kpt[0::3]
            y_ = kpt[1::3]
            v_ = kpt[2::3]
            str_point=' '
            for x, y, v in zip(x_, y_, v_):
                if int(v) > 0:
                    point = str(x/w) +' ' +str(y/h) +' '
                    str_point +=point
                else:
                    point = '0 ' + ' ' + '0 ' + ' '
                    str_point += point
            obj_info =str_box +' ' +str_point +'\n'
            f.write(obj_info)
        print(txt_name)



#
# for name in images_list:
#     img = cv2.imread('/home/py/code/github/yolov5-pose/crowdpose/train/images/' + name['file_name'])
#     vis_id =name['id']
#     kpts, bboxes = get_annokpts(vis_id, annotations_list)
#     txt_name =save_txt_dir +'/' +str(vis_id) +'.txt'
#     json2txt(kpts,bboxes,txt_name,img.shape[1],img.shape[0])





file_name, crowdIndex = find_filename(vis_id, images_list)
kpts, bboxes = get_annokpts(vis_id, annotations_list)
img = cv2.imread('/home/py/code/github/yolov5-pose/crowdpose/train/images/' + file_name)

# img = vis_box(img,bboxes)
# plt.figure(figsize=(12,10))
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()
img=vis_keypoints(img,kpts,crowdIndex)

plt.figure(figsize=(12,10))
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()