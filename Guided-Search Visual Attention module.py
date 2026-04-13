#
# 模拟人类注视，基于saliency和attention计算注视点，生成fixation heatmap

import os
import joblib
import cv2
import numpy as np
from skimage.measure import shannon_entropy
from skimage.filters import threshold_multiotsu
from skimage.feature import graycomatrix, graycoprops
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from skimage.feature import peak_local_max
from scipy.spatial.distance import pdist, squareform
import math
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
from scipy.spatial import distance
from skimage.feature import graycomatrix, graycoprops
from scipy.ndimage import maximum_filter
import matplotlib.pyplot as plt


class ContextAwareVisualAttentionModel:
    def __init__(self, width, height):
        self.width = width
        self.height = height


    def compute_RF_attenion_map(self, lab_image, clf):
        lab_image = cv2.resize(lab_image, (400, 400), interpolation=cv2.INTER_LINEAR)

        # 提取 Lab 通道作为特征
        l_channel = lab_image[:, :, 0]  # L 通道归一化到 [0, 1]
        a_channel = lab_image[:, :, 1]  # a 通道归一化到 [0, 1]
        b_channel = lab_image[:, :, 2]  # b 通道归一化到 [0, 1]

        # 将特征展平为二维数组
        features = np.column_stack([
            l_channel.flatten(),
            a_channel.flatten(),
            b_channel.flatten(),
        ])

        probabilities = clf.predict_proba(features)[:, 1]  # 获取飞机类的概率

        # 将概率值映射到 [0, 255] 范围内
        heatmap = (probabilities * 1).reshape(lab_image.shape[:2]).astype(np.float32)
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.resize(heatmap, (800, 800), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("Attention Map", heatmap)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return heatmap

    def compute_RF_attenion_map_6D(self, lab_image, clf):
        lab_image = cv2.resize(lab_image, (800, 800), interpolation=cv2.INTER_LINEAR)

        def color_contrast_feat(bgr, win=7):
            bgr = cv2.GaussianBlur(bgr, (3, 3), 0.5)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
            # 局部均值
            kernel = np.ones((win, win), np.float32) / (win * win)
            local_mu = cv2.filter2D(lab, -1, kernel, borderType=cv2.BORDER_REFLECT)
            contrast = (lab - local_mu) / (local_mu + 1e-5)
            return np.concatenate([lab, contrast], axis=-1)  # [H,W,6]

        feat = color_contrast_feat(lab_image).reshape(-1, 6)
        prob = clf.predict_proba(feat)[:, 1].reshape(image.shape[:2])
        heatmap = (prob * 1).reshape(lab_image.shape[:2]).astype(np.float32)
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)
        heatmap = cv2.resize(heatmap, (800, 800), interpolation=cv2.INTER_LINEAR)

        return heatmap

    def compute_RF_attenion(self, superpixel_info, clf):
        # 提取 color 信息
        l_values = [pixel['color'][0] for pixel in superpixel_info]
        a_values = [pixel['color'][1] for pixel in superpixel_info]
        b_values = [pixel['color'][2] for pixel in superpixel_info]

        # 将三个一维数组合并成一个二维数组
        features = np.column_stack((l_values, a_values, b_values))

        probabilities = clf.predict_proba(features)[:, 1]  # 获取飞机类的概率
        probabilities_max = np.max(probabilities)
        # 将概率值添加到 superpixel_info 中
        for i, pixel in enumerate(superpixel_info):
            pixel['avg_attention'] = probabilities[i] / probabilities_max

        return superpixel_info

    def object_saliency_map(self, input_image):
        clf = joblib.load('Lab_RF_10_5_b.pkl')
        # clf = joblib.load('gbdt_color6.pkl')
        # t1=time.time()
        resized_image = cv2.resize(input_image, (self.width, self.height))

        object_saliency_map, fixations = self.compute_fixation_heatmap_with_saliency(resized_image, clf)

        return object_saliency_map, fixations

    def compute_attention_map(self, clf, path, name):
        in_path = path + '/' + name
        input_image = cv2.imread(in_path)
        resized_image = cv2.resize(input_image, (self.width, self.height))

        object_saliency_map, fixations = self.compute_fixation_heatmap_with_saliency(resized_image, clf)

        # 将显著性图转换为0-255的范围，以便保存为图像
        object_saliency_map = cv2.GaussianBlur(object_saliency_map, (15, 15), 0)
        object_saliency_map = cv2.normalize(object_saliency_map, None, 0, 255, cv2.NORM_MINMAX)
        # 保存为PNG格式
        cv2.imwrite('./heatmap/fixation_heatmap/' + name, object_saliency_map)

        # 保存注视点信息到txt文件
        # txt_filename = os.path.splitext(name)[0] + '.txt'  # 使用图像名作为txt文件名
        # txt_path = './heatmap/fixation_info/' + txt_filename
        #
        # with open(txt_path, 'w') as f:
        #     # 写入表头
        #     f.write("Position X,Position Y,Max Saliency,Max Attention,Composite Attention,Entropy,PAR\n")
        #
        #     for fixation in fixations:
        #         # 写入每个注视点的信息，使用逗号分隔
        #         f.write(f"{fixation['position'][0]},{fixation['position'][1]},")
        #         f.write(f"{fixation['max_saliency']:.4f},{fixation['max_attention']:.4f},")
        #         f.write(f"{fixation['composite_attention']:.4f},{fixation['entropy']:.4f},")
        #         f.write(f"{fixation['par']:.4f}\n")

    def extract_region_features(self, mask, contour, lab_image, attention_map):

        # 熵计算优化：直接在灰度图上计算
        lab_image_region = cv2.bitwise_and(lab_image, lab_image, mask=mask)
        entropy = shannon_entropy(lab_image_region)


        # 计算注意力最大值
        attention_values = attention_map[mask > 0]
        mean_attention = 0.5 * np.mean(attention_values) + 0.5 * np.max(attention_values) if len(
            attention_values) > 0 else 0.0
        # sum_attention = np.sum(attention_values) if len(attention_values) > 0 else 0.0

        # 周长面积比计算优化：使用外接多边形计算周长
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        par = (perimeter ** 2) / area if area != 0 else 0

        area_weight = min(area / 5000.0, 1.0)  # 假设10000为最大面积阈值

        attention = mean_attention * area_weight
        # 中心点计算优化：直接使用轮廓的矩计算
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"]) if M["m00"] != 0 else 0
        cY = int(M["m01"] / M["m00"]) if M["m00"] != 0 else 0

        # 计算长宽比
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = max(w, h) / min(w, h) if h != 0 else 0

        # 计算矩形度
        bounding_box_area = w * h
        rectangularity = area / bounding_box_area if bounding_box_area != 0 else 0

        return {
            'entropy': entropy,
            'par': par,
            'area': area,
            'max_attention': attention,
            'aspect_ratio': aspect_ratio,
            'rectangularity': rectangularity,
            'position': (cX, cY)

        }

    def seed_mask_create(self, gray_image, attention_map_uint8, thresholds):

        # # 应用直方图均衡化增强对比度
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        # enhanced = clahe.apply(gray_image)
        #
        # # 使用自适应阈值计算低阈值和高阈值
        # median = np.median(enhanced)
        # sigma = 0.33
        # low_threshold = int(max(0, (1.0 - sigma) * median))
        # high_threshold = int(min(255, (1.0 + sigma) * median))
        # print(low_threshold,high_threshold)
        #
        # # 对 gray_image 进行边缘检测
        # edges = cv2.Canny(enhanced, low_threshold, high_threshold)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # edges = cv2.dilate(edges, kernel, iterations=1)
        #
        # # 将边缘部分在 attention_map_uint8 上设为 0，以打断连接区域
        # attention_map_uint8 = cv2.bitwise_and(attention_map_uint8, cv2.bitwise_not(edges))
        # cv2.imshow("attention_map_uint8", attention_map_uint8)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        seed_mask_1 = (attention_map_uint8 > thresholds[0]).astype(np.uint8) * 255
        seed_mask_2 = (attention_map_uint8 > thresholds[1]).astype(np.uint8) * 255
        seed_mask_3 = (attention_map_uint8 > thresholds[2]).astype(np.uint8) * 255

        # dist_transform = cv2.distanceTransform(seed_mask_1, cv2.DIST_L2, 5)
        # _, seed_mask_1 = cv2.threshold(dist_transform, dist_transform.mean(), 255, 0)
        # seed_mask_1 = np.uint8(seed_mask_1)
        #
        # dist_transform = cv2.distanceTransform(seed_mask_2, cv2.DIST_L2, 5)
        # _, seed_mask_2 = cv2.threshold(dist_transform, dist_transform.mean(), 255, 0)
        # seed_mask_2 = np.uint8(seed_mask_2)
        # cv2.imshow("Segmentation1", seed_mask_1)
        # cv2.imshow("Segmentation2", seed_mask_2)
        # cv2.imshow("Segmentation3", seed_mask_3)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 初始化最终种子掩码
        final_seed_mask = np.zeros_like(attention_map_uint8, dtype=np.uint8)
        label_counter = 5  # 用于为最终掩码分配唯一的标签值
        #
        # 第一层：处理 seed_mask_1
        num_labels_1, labels_1 = cv2.connectedComponents(seed_mask_1)
        for label_1 in range(1, num_labels_1):
            component_mask_1 = (labels_1 == label_1).astype(np.uint8) * 255
            # 计算区域面积
            area = np.count_nonzero(component_mask_1)
            # 如果面积过小，跳过
            if area < 100:  # 面积阈值可以根据实际情况调整
                continue
            lab_image_region = cv2.bitwise_and(gray_image, gray_image, mask=component_mask_1)
            entropy = shannon_entropy(lab_image_region)

            if entropy < 0.1:
                # 如果熵小于阈值，直接将该区域添加到最终掩码
                final_seed_mask[component_mask_1 > 0] = label_counter
                label_counter += 1
                # print(entropy)
                # cv2.imshow("Watershed Segmentation", final_seed_mask*255)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
            else:
                # 如果熵较大，使用 seed_mask_2 进一步细化
                second_mask = cv2.bitwise_and(seed_mask_2, seed_mask_2, mask=component_mask_1)
                num_labels_2, labels_2 = cv2.connectedComponents(second_mask)

                for label_2 in range(1, num_labels_2):
                    component_mask_2 = (labels_2 == label_2).astype(np.uint8) * 255
                    # 计算区域面积
                    area = np.count_nonzero(component_mask_2)
                    # 如果面积过小，跳过
                    if area < 50:
                        continue
                    lab_image_region = cv2.bitwise_and(gray_image, gray_image, mask=component_mask_2)
                    entropy = shannon_entropy(lab_image_region)

                    if entropy < 0.1:
                        # 如果熵小于阈值，将该区域添加到最终掩码
                        final_seed_mask[component_mask_2 > 0] = label_counter
                        label_counter += 1
                        # cv2.imshow("Watershed Segmentation", final_seed_mask*255)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    else:
                        # 如果熵仍然较大，使用 seed_mask_3 进一步细化
                        third_mask = cv2.bitwise_and(seed_mask_3, seed_mask_3, mask=component_mask_2)
                        num_labels_3, labels_3 = cv2.connectedComponents(third_mask)

                        for label_3 in range(1, num_labels_3):
                            component_mask_3 = (labels_3 == label_3).astype(np.uint8) * 255
                            # 计算区域面积
                            area = np.count_nonzero(component_mask_3)
                            # 如果面积过小，跳过
                            if area < 20:
                                continue
                            # 直接将第三层的区域添加到最终掩码
                            final_seed_mask[component_mask_3 > 0] = label_counter
                            label_counter += 1
                            # cv2.imshow("Watershed Segmentation", component_mask_3)
                            # cv2.waitKey(0)
                            # cv2.destroyAllWindows()


        return final_seed_mask

    def compute_saliency_mask(self, lab):
        """
        仅返回 high_saliency_mask，不做任何腐蚀/开运算/面积过滤，
        保证后续分水岭可在此 mask 上直接生成种子，不丢失小目标。
        """
        l, a, b = cv2.split(lab)

        # ---------- 1) 三层金字塔 ----------
        lab1 = cv2.pyrDown(lab)
        lab2 = cv2.pyrDown(lab1)
        L0, A0, B0 = l, a, b
        L1, A1, B1 = cv2.split(lab1)
        L2, A2, B2 = cv2.split(lab2)

        # ---------- 2) 每层仅做“稀有度 + 边缘”二值 mask ----------
        def quick_mask(l_ch, a_ch, b_ch):
            # ---------- 1) 细边缘 ----------
            def thin_edge(gray):
                # 1. 梯度幅值
                dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(dx * dx + dy * dy)

                # 2. 多阈值 Otsu（3 类）→ 取最低阈值，保弱边
                mag_uint8 = (mag / (mag.max() + 1e-7) * 255).astype(np.uint8)
                thr_multi = threshold_multiotsu(mag_uint8, classes=3)
                low_thr = thr_multi[0]

                # 3. 细边缘二值图
                thin = (mag_uint8 > low_thr).astype(np.uint8) * 255

                # 4. 轻度开运算去毛刺，核极小防止膨胀
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
                thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN, kernel, 1)
                return thin, mag_uint8

            edge_bin, mag = thin_edge(l_ch)

            # ---------- 2) AB 聚类 → Soft 稀有度 ----------
            # 2) 二维直方图稀有度（无聚类，无颗粒）
            h, w = l_ch.shape
            ab = np.stack([a_ch.ravel(), b_ch.ravel()], axis=1).astype(np.float32)

            # 2-1 量化到 64×64  bins（可调）
            bins = 128
            a_min, a_max = ab[:, 0].min(), ab[:, 0].max()
            b_min, b_max = ab[:, 1].min(), ab[:, 1].max()
            # 避免除 0
            a_range = a_max - a_min + 1e-7
            b_range = b_max - b_min + 1e-7

            a_idx = ((ab[:, 0] - a_min) / a_range * (bins - 1)).astype(np.int32)
            b_idx = ((ab[:, 1] - b_min) / b_range * (bins - 1)).astype(np.int32)

            # 2-2 统计频次
            hist = np.zeros((bins, bins), dtype=np.int32)
            np.add.at(hist, (a_idx, b_idx), 1)

            # 2-3 相对稀有度：把频次转为「分位」→ 目标落在高分位
            hist_smooth = cv2.GaussianBlur(hist.astype(np.float32), (9, 9), 0)
            # 计算分位（0=最频繁，1=最稀疏）
            rank = np.argsort(np.argsort(hist_smooth.ravel())).reshape(hist_smooth.shape)
            rank = rank.astype(np.float32) / rank.max()  # 0~1
            # 反转：高分位 → 高稀有度
            rarity_hist = 1.0 - rank

            # 2-4 反投影 → 连续稀有度图
            rarity_prob = rarity_hist[a_idx, b_idx]
            rarity_map = rarity_prob.reshape(h, w)
            rarity_map = cv2.normalize(rarity_map, None, 0, 1, cv2.NORM_MINMAX)
            rarity_uint8 = (rarity_map * 255).astype(np.uint8)
            # rarity_uint8 = cv2.bilateralFilter(rarity_uint8, d=11, sigmaColor=10, sigmaSpace=10)
            # cv2.imshow('edge_bin', edge_bin)
            # cv2.imshow('mag', mag)
            # cv2.imshow('rarity_uint8', rarity_uint8)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # ---------- 3) 稀有度多阈值 Otsu ----------
            thr_multi = threshold_multiotsu(rarity_uint8, classes=4)
            low_thr = thr_multi[0]  # 最低段 → 前景

            # 4-1 稀有度「肉」
            mask_rarity_uint8 = (rarity_uint8 > low_thr).astype(np.uint8) * 255
            dist_transform = cv2.distanceTransform(mask_rarity_uint8, cv2.DIST_L2, 5)
            _, seed_mask_1 = cv2.threshold(dist_transform, 2, 255, 0)
            mask_rarity_uint8 = np.uint8(seed_mask_1)

            # ---------- 4) 交集 + 开闭修剪 ----------
            # 4-1 边缘与稀有度取交集 → 带边界的面
            # edge_bin = thin_edge(l_ch)  # 1 px 骨架
            mask_face = cv2.bitwise_or(mask_rarity_uint8, edge_bin)

            # 4-2 先腐蚀掉毛刺 & 断开细小粘连
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            mask_clean = cv2.erode(mask_face, kernel, iterations=1)

            # 4-3 再膨胀补回真实边缘 + 填小空洞
            mask_clean = cv2.dilate(mask_clean, kernel, iterations=2)
            # cv2.imshow('mask_clean', mask_clean)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            return mask_clean

        bin0 = quick_mask(L0, A0, B0)
        bin1 = quick_mask(L1, A1, B1)
        bin2 = quick_mask(L2, A2, B2)

        # ---------- 3) 上采到 800×800 ----------
        bin1 = cv2.resize(bin1, (800, 800), interpolation=cv2.INTER_NEAREST)
        bin2 = cv2.resize(bin2, (800, 800), interpolation=cv2.INTER_NEAREST)

        # ---------- 2. 级联融合 ----------
        # (1) 粗→中：粗 mask 做“母体”，中 mask 做“修正”
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bin2_closed = cv2.morphologyEx(bin2, cv2.MORPH_CLOSE, kernel_close, iterations=2)  # 填大洞
        body = cv2.bitwise_and(bin1, bin2_closed)  # 保留粗尺度支持下的中等区域

        # (2) 细→中：细 mask 提供锐利边缘
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bin0_open = cv2.morphologyEx(bin0, cv2.MORPH_OPEN, kernel_open, iterations=1)  # 去毛刺
        edge = cv2.bitwise_or(bin0_open, body)
        # 闭运算只做大洞填补，核变小
        kernel_final = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        edge_closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel_final, iterations=1)

        # ---------- 3. 填内部孔洞 + 小面积过滤 ----------
        contours, _ = cv2.findContours(edge_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        high_saliency_mask = np.zeros_like(edge_closed)
        for cnt in contours:
            if cv2.contourArea(cnt) >= 100:  # 与后续过滤保持一致
                cv2.drawContours(high_saliency_mask, [cnt], -1, 255, thickness=cv2.FILLED)

        return

    def compute_saliency_map(self, lab):
        """
        【修改版】直接输出 连续灰度显著性热力图 (0~255)
        无任何二值化、无mask、无形态学，纯Heatmap
        """
        l, a, b = cv2.split(lab)

        # ---------- 1) 三层金字塔 ----------
        lab1 = cv2.pyrDown(lab)
        lab2 = cv2.pyrDown(lab1)
        L0, A0, B0 = l, a, b
        L1, A1, B1 = cv2.split(lab1)
        L2, A2, B2 = cv2.split(lab2)

        # ---------- 2) 每层输出【连续热力图】，不再输出mask ----------
        def quick_heatmap(l_ch, a_ch, b_ch):
            # ---------- 1) 细边缘梯度幅值（连续值，非二值） ----------
            def edge_magnitude(gray):
                dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
                dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
                mag = np.sqrt(dx * dx + dy * dy)
                mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
                return mag

            edge_map = edge_magnitude(l_ch)

            # ---------- 2) AB 颜色稀有度（连续值，核心Heatmap） ----------
            h, w = l_ch.shape
            ab = np.stack([a_ch.ravel(), b_ch.ravel()], axis=1).astype(np.float32)
            bins = 128
            a_min, a_max = ab[:, 0].min(), ab[:, 0].max()
            b_min, b_max = ab[:, 1].min(), ab[:, 1].max()
            a_range = a_max - a_min + 1e-7
            b_range = b_max - b_min + 1e-7

            a_idx = ((ab[:, 0] - a_min) / a_range * (bins - 1)).astype(np.int32)
            b_idx = ((ab[:, 1] - b_min) / b_range * (bins - 1)).astype(np.int32)

            hist = np.zeros((bins, bins), dtype=np.int32)
            np.add.at(hist, (a_idx, b_idx), 1)
            hist_smooth = cv2.GaussianBlur(hist.astype(np.float32), (9, 9), 0)

            rank = np.argsort(np.argsort(hist_smooth.ravel())).reshape(hist_smooth.shape)
            rank = rank.astype(np.float32) / rank.max()
            rarity_hist = 1.0 - rank  # 稀有度：越高越显著

            rarity_map = rarity_hist[a_idx, b_idx].reshape(h, w)
            rarity_map = cv2.normalize(rarity_map, None, 0, 1, cv2.NORM_MINMAX)

            # ---------- 3) 融合：边缘 + 稀有度 = 连续显著性热力图 ----------
            saliency_heat = cv2.addWeighted(rarity_map, 0.7, edge_map, 0.3, 0)
            return saliency_heat

        # 多尺度连续热力图
        heat0 = quick_heatmap(L0, A0, B0)
        heat1 = quick_heatmap(L1, A1, B1)
        heat2 = quick_heatmap(L2, A2, B2)

        # 上采样到统一尺寸
        heat1 = cv2.resize(heat1, (800, 800), interpolation=cv2.INTER_LINEAR)
        heat2 = cv2.resize(heat2, (800, 800), interpolation=cv2.INTER_LINEAR)

        # ---------- 多尺度融合（连续值加权） ----------
        fused_heat = heat0 * 0.4 + heat1 * 0.3 + heat2 * 0.3
        fused_heat = cv2.normalize(fused_heat, None, 0, 1, cv2.NORM_MINMAX)

        # 高斯模糊，让热力图更平滑
        fused_heat = cv2.GaussianBlur(fused_heat, (7, 7), 1.0)

        # 输出 0~255 连续灰度热力图
        return (fused_heat * 255).astype(np.uint8)

    def compute_fixation_heatmap_with_saliency(self, input_image, clf):

        input_image = cv2.pyrMeanShiftFiltering(input_image, 25, 25)

        # 提取外轮廓
        lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)
        high_saliency_mask = self.compute_saliency_mask(lab_image)
        contours, _ = cv2.findContours(high_saliency_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 将输入图像转换为LAB色彩空间
        # lab_image_mask = cv2.bitwise_and(lab_image, lab_image, mask=high_saliency_mask)
        # hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        attention_map = self.compute_RF_attenion_map(lab_image, clf)
        # attention_map = self.compute_RF_attenion_map_6D(lab_image, clf)
        attention_map_uint8 = (attention_map * 255).astype(np.uint8)
        # cv2.imshow('attention_map', attention_map_uint8)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        thresholds = threshold_multiotsu(attention_map_uint8, nbins=256, classes=4)
        # 提取种子区域（最高两层）

        # 处理每个轮廓区域
        all_regions = []

        # visualization = np.zeros_like(input_image)
        # # 为每个mask分配一个随机颜色
        # for i, mask in enumerate(low_entropy_regions):
        #     # 生成随机颜色
        #     color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        #     # 将mask应用到可视化图像上
        #     visualization[mask > 0] = color
        # # 显示结果
        # cv2.imshow("Low Entropy Regions Visualization", visualization)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 创建一个与原图大小相同的空白掩码
        filled_mask = np.zeros_like(high_saliency_mask)

        # 填充外轮廓以获取完整的区域
        for contour in contours:
            cv2.drawContours(filled_mask, [contour], -1, 255, thickness=cv2.FILLED)

        gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        high_entropy_attention_map = cv2.bitwise_and(attention_map_uint8, attention_map_uint8, mask=filled_mask)
        markers = self.seed_mask_create(gray_image, high_entropy_attention_map, thresholds)
        # 将背景标记为 0

        # cv2.imshow("filled_mask", filled_mask)
        # # cv2.imshow("Watershed Segmentation with Random Colors", markers)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        markers[filled_mask == 0] = 1

        # # 获取区域数量，包括背景
        # num_regions = np.max(markers) + 1
        # # 生成随机颜色
        # colors = np.random.randint(0, 256, (num_regions, 3), dtype=np.uint8)
        # colors[1] = [0, 0, 0]  # 将背景颜色设置为黑色
        # colors[0] = [128, 128, 128]  # 将背景颜色设置为黑色
        # # 创建可视化图像
        # visualization = colors[markers]
        # # 将 visualization 叠加到原图上
        # overlay = cv2.addWeighted(image, 0.3, visualization, 0.7, 0)
        # # 显示结果
        # cv2.imshow("Watershed Segmentation with Random Colors", overlay)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 确保输入图像和标记矩阵的格式正确
        if input_image.dtype != np.uint8:
            input_image = input_image.astype(np.uint8)
        if markers.dtype != np.int32:
            markers = markers.astype(np.int32)

        # meanShift_image = cv2.pyrMeanShiftFiltering(lab_image,20,50)

        # 应用Watershed算法
        markers = cv2.watershed(input_image, markers)

        # # 获取区域数量，包括背景
        # num_regions = np.max(markers) + 1
        # # 生成随机颜色
        # colors = np.random.randint(0, 256, (num_regions, 3), dtype=np.uint8)
        # colors[1] = [0, 0, 0]  # 将背景颜色设置为黑色
        # # 创建可视化图像
        # visualization = colors[markers]
        # overlay = cv2.addWeighted(image, 0.3, visualization, 0.7, 0)
        # # 显示结果
        # cv2.imshow("Watershed Segmentation with Random Colors", overlay)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 提取分割结果
        for label in np.unique(markers):
            if label == -1:  # 忽略边界
                continue
            if label == 1:  # 跳过背景
                continue
            segment_mask = ((markers == label) * 255).astype(np.uint8)
            contours, _ = cv2.findContours(segment_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            contour = contours[0]  # 取最大的轮廓
            region_features = self.extract_region_features(segment_mask, contour, lab_image, attention_map_uint8)
            #print(region_features['aspect_ratio'],region_features['rectangularity'],region_features['par'])
            # 宽松条件：排除明显不是飞机的区域
            if region_features['aspect_ratio']>3.0:  # 太细长，不像飞机
                continue
            elif region_features['rectangularity'] < 0.2 or region_features['rectangularity'] > 0.9:  # 太不规则或太矩形，不像飞机
                continue
            elif region_features['par'] < 30:  # 边缘太复杂，不像飞机
                continue
            else:
                #low_entropy_regions.append(segment_mask)
                all_regions.append(region_features)
            # all_regions.append(region_features)

        # visualization = np.zeros_like(input_image)  # 创建一个与输入图像大小相同的空白图像
        # # 为每个mask分配一个随机颜色
        # for i, mask in enumerate(low_entropy_regions):
        #     # 生成随机颜色
        #     color = (np.random.randint(0, 256), np.random.randint(0, 256), np.random.randint(0, 256))
        #     # 将mask应用到可视化图像上
        #     visualization[mask > 0] = color
        #
        # # 显示结果
        # cv2.imshow("Low Entropy Regions Visualization", visualization)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # 计算综合显著性并排序0.3 * region['mean_saliency'] + 0.7 *
        # fixations_contral = self.merge_fixations(all_regions[:k])
        fixations_contral = all_regions

        # 生成热图
        heatmap = self.generate_weighted_heatmap(fixations_contral, input_image.shape[:2])

        return heatmap, fixations_contral



    def generate_weighted_heatmap(self, fixations, image_shape):

        def compute_sigma_from_area(area, min_area=200, max_area=10000, min_sigma=25, max_sigma=80):
            """
            根据区域面积自适应计算高斯核的 sigma
            面积越小，sigma 越小；面积越大，sigma 越大
            """
            area = np.clip(area, min_area, max_area)
            log_area = np.log(area)
            log_min = np.log(min_area)
            log_max = np.log(max_area)
            sigma = min_sigma + (max_sigma - min_sigma) * (log_area - log_min) / (log_max - log_min)
            return int(round(sigma))
        # 提取 composite_score 并归一化
        composite_attention = [fixation['max_attention'] for fixation in fixations]

        min_composite = min(composite_attention)
        max_composite = max(composite_attention)

        scores = [(score - min_composite) / (max_composite - min_composite + 0.00001) for score in composite_attention]

        # Create an empty heatmap
        heatmap = np.zeros(image_shape, dtype=np.float32)

        # Apply Gaussian kernel density estimation with weights
        for i, fixation in enumerate(fixations):
            x, y = fixation['position']
            area = fixation['area']
            sigma = 50#compute_sigma_from_area(area)
            weight = scores[i]

            # Ensure the fixation coordinates are within the image boundaries
            if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
                # Create a Gaussian kernel centered at (x, y)
                kernel = np.exp(
                    -((np.arange(image_shape[0])[:, None] - y) ** 2 + (np.arange(image_shape[1])[None, :] - x) ** 2) / (
                            2 * sigma ** 2))
                kernel /= kernel.sum()  # Normalize the kernel

                # Add the weighted kernel to the heatmap
                heatmap += kernel * weight

        # Normalize the heatmap to the range [0, 1]
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max() \
 \
                # 计算所有注视点的中心
        if fixations:
            center_x = int(np.mean([fixation['position'][0] for fixation in fixations]))
            center_y = int(np.mean([fixation['position'][1] for fixation in fixations]))

            # 创建一个以注视点中心为中心的高斯核
            x_coords = np.arange(image_shape[1])
            y_coords = np.arange(image_shape[0])
            X, Y = np.meshgrid(x_coords, y_coords)
            gaussian_kernel = np.exp(-((X - center_x) ** 2 + (Y - center_y) ** 2) / (2 * 400 ** 2))
            gaussian_kernel /= gaussian_kernel.max()

            # 将高斯核应用到热图上heatmap *
            heatmap = heatmap * gaussian_kernel

            # 归一化热图
            heatmap = cv2.normalize(heatmap, None, 0, 1, cv2.NORM_MINMAX)

        return heatmap

    def generate_dual_channel_separate(
            self,
            input_image,
            image_filename,
            clf,
            saliency_dir="./saliency_channel",
            attention_dir="./attention_channel"
    ):
        """
        最终版：两个通道均输出【纯灰度连续Heatmap】
        - 显著性通道：compute_saliency_mask 直接输出连续热图
        - 注意力通道：归一化灰度热图
        无彩色、无二值、纯Heatmap
        """
        import os
        os.makedirs(saliency_dir, exist_ok=True)
        os.makedirs(attention_dir, exist_ok=True)

        lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB)

        # ========================
        # 通道1：显著性 → 直接输出连续灰度Heatmap
        # ========================
        saliency_heatmap = self.compute_saliency_map(lab_image)

        # ========================
        # 通道2：注意力 → 灰度Heatmap
        # ========================
        attention_map = self.compute_RF_attenion_map(lab_image, clf)
        attention_heatmap = cv2.normalize(attention_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # ========================
        # 保存纯灰度图（无任何彩色）
        # ========================
        saliency_save_path = os.path.join(saliency_dir, image_filename)
        attention_save_path = os.path.join(attention_dir, image_filename)

        cv2.imwrite(saliency_save_path, saliency_heatmap)
        cv2.imwrite(attention_save_path, attention_heatmap)

        print(f"✅ 保存完成：")
        print(f"   显著性灰度Heatmap：{saliency_save_path}")
        print(f"   注意力灰度Heatmap：{attention_save_path}")

        return saliency_heatmap, attention_heatmap


if __name__ == "__main__":

    raw_data_path = 'E:/object detection/RSOD/Stimuli_ob'
    file_names = os.listdir(raw_data_path)
    model = ContextAwareVisualAttentionModel(800, 800)
    clf = joblib.load('Lab_RF_10_5_b.pkl')
    #clf = joblib.load('gbdt_color6.pkl')

    # warnings.filterwarnings("ignore")
    # image = cv2.imread("./data/00114.jpg")
    # saliency_map, fixations_contral = model.object_saliency_map(image)
    # gaze_points_image = image.copy()
    # for point in fixations_contral:
    #     # 绘制注视点
    #     cv2.circle(gaze_points_image, point['position'], 5, (0, 0, 255), -1)
    #
    #     # 提取需要显示的特征值
    #     max_attention = point['aspect_ratio']
    #     rectangularity = point['rectangularity']
    #     par = point['par']
    #     # print(max_saliency,max_attention)
    #     # composite_saliency = point['composite_saliency']
    #
    #     # 将特征值转换为字符串f"s: {composite_saliency:.2f}\n"
    #     # text = f"s: {max_saliency:.2f}\n" \
    #     #        f"a: {avg_attention:.2f}\n" \
    #     #        f"p: {par:.2f}\n" \
    #     #        f"e: {entropy:.2f}"
    #     text = f"a: {par:.2f}"
    #
    #     # 在注视点旁边绘制文本
    #     text_x = point['position'][0] - 100
    #     text_y = point['position'][1] - 10
    #     cv2.putText(gaze_points_image, text, (text_x, text_y),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    #
    # cv2.imshow("Gaze Points with Features", gaze_points_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # cv2.imshow("Context-Aware Saliency Map", saliency_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # for name in file_names:
    #     print(name)
    #     model.compute_attention_map(clf,raw_data_path,name)
    saliency_save_dir = "./heatmap/saliency_channel"  # 显著性通道保存文件夹
    attention_save_dir = "./heatmap/attention_channel"  # 注意力通道保存文件夹
    model_width = 800  # 模型输入宽度
    model_height = 800  # 模型输入高度

    # 支持的图片格式（可根据需要扩展）
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    file_names = [
        f for f in os.listdir(raw_data_path)
        if f.lower().endswith(image_extensions)
    ]

    if not file_names:
        print(f"⚠️ 在 {raw_data_path} 中未找到图片文件！")
    else:
        print(f"📌 共找到 {len(file_names)} 张图片，开始处理...")

    # ======================== 批量处理图片 ========================
    for idx, image_filename in enumerate(file_names, 1):
        try:
            # 拼接完整图片路径
            image_path = os.path.join(raw_data_path, image_filename)
            # 读取图片
            input_image = cv2.imread(image_path)

            if input_image is None:
                print(f"❌ 第 {idx} 张图片 {image_filename} 读取失败，跳过")
                continue

            # 调用双通道保存方法
            saliency_heatmap, attention_heatmap = model.generate_dual_channel_separate(
                input_image=input_image,
                image_filename=image_filename,
                clf=clf,
                saliency_dir=saliency_save_dir,
                attention_dir=attention_save_dir
            )

            print(f"✅ 第 {idx}/{len(file_names)} 张处理完成：{image_filename}")

        except Exception as e:
            print(f"❌ 第 {idx} 张图片 {image_filename} 处理出错：{str(e)}")
            continue

    print("\n🎉 批量处理完成！")
    print(f"📁 显著性通道保存路径：{os.path.abspath(saliency_save_dir)}")
    print(f"📁 注意力通道保存路径：{os.path.abspath(attention_save_dir)}")
