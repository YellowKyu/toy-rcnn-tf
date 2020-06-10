import cv2
import sys
import random
import numpy as np


class DataManager(object):

    def __init__(self):
        pass

    def gen_triangle(self, shape):
        max_y = shape[0]
        max_x = shape[1]

        # gen three points
        pt1 = [random.randint(0, max_x), random.randint(0, max_y)]
        pt2 = [random.randint(0, max_x), random.randint(0, max_y)]
        pt3 = [random.randint(0, max_x), random.randint(0, max_y)]
        pts = [pt1, pt2, pt3]

        # gen triangle's rectangular bounding box
        x1, x2, y1, y2 = pt1[0], pt1[0], pt1[1], pt1[1]
        for pt in pts:
            x1 = min(pt[0], x1)
            x2 = max(pt[0], x2)
            y1 = min(pt[1], y1)
            y2 = max(pt[1], y2)

        return np.array(pts), np.array([x1, y1, x2, y2])

    def gen_circle(self, shape):
        max_y = shape[0]
        max_x = shape[1]
        quarter_y = int(max_y / 3)

        # gen circle's center and radius
        ctr_x = random.randint(0, max_x)
        ctr_y = random.randint(0, max_y)
        radius = random.randint(0, quarter_y)

        # gen cercle's rectangular bounding box
        # min/max to make box inside (0, max_x) and (0, max_y)
        x1 = max(0, ctr_x - radius)
        y1 = max(0, ctr_y - radius)
        x2 = min(ctr_x + radius, max_x - 1)
        y2 = min(ctr_y + radius, max_y - 1)

        return ctr_x, ctr_y, radius, np.array([x1, y1, x2, y2]).astype("float32")

    def gen_random_shape(self, image, shape):
        shape_choice = random.randint(1, 2)
        color = (random.randint(0, 254), random.randint(0, 254), random.randint(0, 254))

        # if shape_choice == 1:
        #     pts, box = self.gen_triangle(shape)
        #     cv2.drawContours(image, [pts], 0, color, -1)
        # elif shape_choice == 2:
        ctr_x, ctr_y, radius, box = self.gen_circle(shape)
        cv2.circle(image, (ctr_x, ctr_y), radius, color, -1)

        return image, box, shape_choice

    def draw_objectness_mask(self, box, mask, center_ratio=1.0):
        w = box[2] - box[0]
        h = box[3] - box[1]
        ctr_x = box[0] + (w / 2.0)
        ctr_y = box[1] + (h / 2.0)
        scaled_w = (w * center_ratio) / 2.0
        scaled_h = (h * center_ratio) / 2.0

        mask = cv2.rectangle(mask, (int(ctr_x - scaled_w), int(ctr_y - scaled_h)),
                             (int(ctr_x + scaled_w), int(ctr_y + scaled_h)), (255), -1)
        return mask

    def draw_tlbr_mask(self, box, shape, t_m, l_m, b_m, r_m):
        w = box[2] - box[0]
        h = box[3] - box[1]
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        for x in range(int(x1), int(x2)):
            for y in range(int(y1), int(y2)):
                t_m[y, x] = (y - y1)
                l_m[y, x] = (x - x1)
                b_m[y, x] = (y2 - y)
                r_m[y, x] = (x2 - x)
        return t_m, l_m, b_m, r_m

    def draw_xyxy_mask(self, box, shape, x1_m, y1_m, x2_m, y2_m):
        x1 = float(box[0]) / float(shape[1])
        y1 = float(box[1]) / float(shape[0])
        x2 = float(box[2]) / float(shape[1])
        y2 = float(box[3]) / float(shape[0])
        x1_m[box[1]:box[3], box[0]:box[2]] = x1
        y1_m[box[1]:box[3], box[0]:box[2]] = y1
        x2_m[box[1]:box[3], box[0]:box[2]] = x2
        y2_m[box[1]:box[3], box[0]:box[2]] = y2
        return x1_m, y1_m, x2_m, y2_m


    def gen_toy_detection_sample(self, num, shape=(108, 192, 3), need_mask_label=False):
        x = []
        bboxes = []
        categories = []
        objectness_mask_labels = []
        bboxes_mask_labels = []

        for _ in range(num):
            image = np.zeros(shape, np.uint8)
            objectness_mask_label = np.zeros((shape[0], shape[1], 1), np.uint8)
            x1_m_l = np.zeros((shape[0], shape[1]), np.float32)
            y1_m_l = np.zeros((shape[0], shape[1]), np.float32)
            x2_m_l = np.zeros((shape[0], shape[1]), np.float32)
            y2_m_l = np.zeros((shape[0], shape[1]), np.float32)

            box_list = []
            cat_list = []
            num_obj = random.randint(1, 3)
            for _ in range(num_obj):
                image, box, category = self.gen_random_shape(image, shape)
                box_list.append(np.array(box))
                cat_list.append(category)
                if need_mask_label:
                    objectness_mask_label = self.draw_objectness_mask(box, objectness_mask_label)
                    # x1_m_l, y1_m_l, x2_m_l, y2_m_l = self.draw_xyxy_mask(box, shape, x1_m_l, y1_m_l, x2_m_l, y2_m_l)
                    x1_m_l, y1_m_l, x2_m_l, y2_m_l = self.draw_tlbr_mask(box, shape, x1_m_l, y1_m_l, x2_m_l, y2_m_l)

            for _ in range(5 - num_obj):
                box_list.append(np.array([0, 0, 0, 0]))

            x.append(image)
            bboxes.append(np.array(box_list))
            categories.append(cat_list)
            objectness_mask_labels.append(objectness_mask_label)

            xyxy_mask_label = np.stack([x1_m_l, y1_m_l, x2_m_l, y2_m_l], axis=-1)
            bboxes_mask_labels.append(xyxy_mask_label)

        return np.asarray(x), np.array(bboxes), np.asarray(categories), np.asarray(objectness_mask_labels), np.asarray(bboxes_mask_labels)

    def gen_toy_detection_datasets(self, train_size=300, test_size=100):
        train_x, train_y, train_cat, train_mask, train_bbox_mask = self.gen_toy_detection_sample(train_size, need_mask_label=True)
        test_x, test_y, test_cat, test_mask, test_bbox_mask = self.gen_toy_detection_sample(test_size, need_mask_label=True)
        return train_x, train_y, train_cat, train_mask, train_bbox_mask, test_x, test_y, test_cat, test_mask, test_bbox_mask

    def gen_mnsit_detection_datasets(self, mnist_path, train_size=300, test_size=100):
        pass


if __name__ == '__main__':

    my_manager = DataManager()
    train_x, train_y, train_cat, train_mask, train_bbox_mask, test_x, test_y, test_cat, test_mask, test_bbox_mask = my_manager.gen_toy_detection_datasets()
    print(train_x.shape, train_y.shape, train_cat.shape, train_mask.shape, train_bbox_mask.shape)
    print(test_x.shape, test_y.shape, test_cat.shape, test_mask.shape, test_bbox_mask.shape)

    bbox = train_y[0][0]
    map = train_bbox_mask[0]
    print(train_y[0])
    print(bbox, map[int((bbox[1] + bbox[3]) / 2)][int((bbox[0] + bbox[2]) / 2)])
    for box in train_y[0]:
        train_x[0] = cv2.rectangle(train_x[0], (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 1)

    cv2.imshow('image', train_x[0])
    cv2.imshow('mask', train_mask[0])
    cv2.imshow('bb_mask_x1', cv2.applyColorMap(train_bbox_mask[0][:, :, 0].astype(np.uint8), cv2.COLORMAP_JET))
    cv2.imshow('bb_mask_y1', cv2.applyColorMap(train_bbox_mask[0][:, :, 1].astype(np.uint8), cv2.COLORMAP_JET))
    cv2.imshow('bb_mask_x2', cv2.applyColorMap(train_bbox_mask[0][:, :, 2].astype(np.uint8), cv2.COLORMAP_JET))
    cv2.imshow('bb_mask_y2', cv2.applyColorMap(train_bbox_mask[0][:, :, 3].astype(np.uint8), cv2.COLORMAP_JET))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
