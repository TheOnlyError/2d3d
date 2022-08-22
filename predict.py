import logging
import os
import time

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np

from training.post_process import post_process
from training.utils import ind2rgb, ind2rgba, rgb2ind
from datasets import floorplans

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
logging.disable(logging.WARNING)

from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def pad_to_old(x, stride):
    h, w = x.shape[:2]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = tf.constant([[lh, uh], [lw, uw], [0, 0]])

    # zero-padding by default.
    out = tf.pad(x, pads, "CONSTANT", 255)

    return out, pads


def pad_to(x, stride):
    h, w = x.shape[:2]

    if h % stride > 0:
        new_h = h + stride - h % stride
    else:
        new_h = h
    if w % stride > 0:
        new_w = w + stride - w % stride
    else:
        new_w = w

    new_h = max(new_h, new_w)
    new_w = max(new_h, new_w)
    lh, uh = int((new_h - h) / 2), int(new_h - h) - int((new_h - h) / 2)
    lw, uw = int((new_w - w) / 2), int(new_w - w) - int((new_w - w) / 2)
    pads = tf.constant([[lh, uh], [lw, uw], [0, 0]])

    # zero-padding by default.
    out = tf.pad(x, pads, "CONSTANT", 255)

    return out, pads


def unpad(x, pad):
    if pad[2] + pad[3] > 0:
        x = x[:, :, pad[2]:-pad[3], :]
    if pad[0] + pad[1] > 0:
        x = x[:, :, :, pad[0]:-pad[1]]
    return x


def plot_diff(image_org):
    classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
    classes = ['bg'] + classes
    datasets = [
        'multi_plans_test_augment'
        # 'cubicasa5k_augment',
        # 'r3d_augment'
    ]

    test_dataset = floorplans.load_test_data(classes, datasets, normalize=False)

    plt.figure(dpi=300)
    plt.imshow(image_org)
    plt.axis('off')
    plt.show()

    plt.figure(dpi=400)
    for (image, label) in test_dataset.skip(2).take(1).batch(1):
        plt.imshow(image[0].numpy().astype(np.uint8))
        # plt.imshow(np.array(image[0]).astype(np.uint8))
    plt.axis('off')
    plt.show()
    cv2.imwrite('imgtest.png', image[0].numpy())

    exit(0)


# files: [file.jpg or file.png]
def predict(files, tta=False, overlay=False):
    # unet_model = tf.keras.models.load_model('models/r3d_augment', compile=False)
    # unet_model = tf.keras.models.load_model('models/zeng_new', compile=False)
    # unet_model = tf.keras.models.load_model('models/zeng_final3', compile=False)
    # unet_model = tf.keras.models.load_model('models/zeng_final', compile=False)
    # unet_model = tf.keras.models.load_model('models/multi_final', compile=False)

    unet_model = tf.keras.models.load_model('models/cubi_single_zeng1', compile=False)

    # backbone = 'resnet50'
    backbone = 'efficientnetb2'
    # backbone = 'vgg16'
    normalize = backbone[0] != 'e'
    print("Backbone: {0}, normalize: {1}".format(backbone, normalize))

    predictions = []
    images = []
    for file in files:
        images.append(cv2.imread('resources/' + file))
    for i, image_org in enumerate(images):
        # image_org = cv2.resize(image_org, [512, 512])
        image_org, pads = pad_to(image_org, 32)

        image = tf.convert_to_tensor(image_org, dtype=tf.uint8)
        image = tf.cast(image, dtype=tf.float32)

        results = []
        rots = 4 if tta else 1
        for k in range(rots):
            shp = image.shape

            if normalize:
                image_batch = tf.reshape(image, [-1, shp[0], shp[1], 3]) / 255  # Only divide if not effnet
            else:
                image_batch = tf.reshape(image, [-1, shp[0], shp[1], 3])

            prediction = unet_model.predict_on_batch(image_batch)
            if isinstance(prediction, tuple):
                prediction = prediction[1]
            result = prediction[0].argmax(axis=-1)
            results.append(np.rot90(result, k=-k))
            image = tf.image.rot90(image, k=1)

        result = np.mean(results, axis=0)
        result_pp = post_process(result)

        # Unpad result
        # TODO
        timestr = time.strftime("%Y%m%d-%H%M%S")
        if overlay:
            plt.figure(dpi=400)
            plt.imshow(image_org)
            plt.imshow(ind2rgb(result), alpha=0.7)
            plt.axis('off')
            plt.savefig('results/' + timestr, bbox_inches='tight', pad_inches=0)
            plt.show()
        mpimg.imsave("results/result" + timestr + str(i) + '.png', ind2rgb(result))
        mpimg.imsave("results/result_pp" + timestr + str(i) + '.png', ind2rgb(result_pp))
        predictions.append(result_pp)
    return predictions


def write(file, model_name, img):
    path = 'results/' + model_name + '/'
    if not os.path.exists(path):
        os.makedirs(path)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    cv2.imwrite(path + file + '_' + timestr + '.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(path + file + '_' + timestr + '.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main():
    # R3D
    models = [
        # ['zeng_zeng_r3d_final_vgg16_32,64,128,256,512_r3d_augment_20220809-184713', True],
        # ['cubicasa5k_cubicasa5k_r3d_full_vgg16_32,64,128,256,512_r3d_augment_20220814-165742', True],
        # ['unetpp_r3d_full_efficientnetb2_32,64,128,256,512_r3d_augment_20220815-145449', False],
        ['cab2_map_abc_r3d_EfficientNetB2_32,64,128,256,512_r3d_augment_20220809-140528', False],
    ]

    # CubiCasa
    # models = [
    #     ['zeng_zeng_cubi_final_vgg16_64,128,256,512,1024_cubicasa5k_augment_20220725-194943', True],
    #     ['cubicasa5k_cubicasa5k_cubi_full_vgg16_32,64,128,256,512_cubicasa5k_augment_20220814-181509', True],
    #     ['cab2_map_abc_cubi_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220811-200057', False],
    # ]

    # Multi
    # models = [
    #     ['zeng_zeng_multi_final_vgg16_32,64,128,256,512_multi_plans_augment_20220728-135338', True],
    #     ['cubicasa5k_cubicasa5k_multi_full_vgg16_32,64,128,256,512_multi_plans_augment_20220814-141054', True],
    #     ['cab2_map_abc_single_sam_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220819-001613', False],
    # ]

    # models = [
    #     # ['unet_cubi_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220816-081929', False],
    #     ['unet_combi2_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220820-083922', False]
    #     # [
    #     #     'cab2_map_abc_single_sam_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220819-001613',
    #     #     False]
    # ]

    # unet_model = tf.keras.models.load_model('models/ours_multi_EfficientNetB2_32,64,128,256,512_r3d_augment_20220708-220332', compile=False)

    # unet_model = tf.keras.models.load_model('models/cubi_zeng1', compile=False)
    # unet_model = tf.keras.models.load_model('models/cubi_cubi1', compile=False)
    # unet_model = tf.keras.models.load_model('models/cubi_unetpp1', compile=False)
    # unet_model = tf.keras.models.load_model('models/cubi_unetpp_20220626-154449', compile=False)
    # unet_model = tf.keras.models.load_model('models/unetpp_efficientnetb2_32,64,128,256,512_cubicasa5k_augment_20220629-180526', compile=False)
    # unet_model = tf.keras.models.load_model('models/cubi_unet3p_20220622-193802', compile=False)
    # unet_model = tf.keras.models.load_model('models/cubi_unet3p_20220623-013908', compile=False)
    # unet_model = tf.keras.models.load_model('models/cubi_unet3p_20220625-200242', compile=False)

    # unet_model = tf.keras.models.load_model('models/unetpp_efficientnetb2_32,64,128,256,512_cubicasa5k_augment_20220708-105101', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p__32,64,128,256,512_cubicasa5k_augment_20220708-155708', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220708-191618', compile=False)

    # unet_model = tf.keras.models.load_model('models/unet3p_3aaf_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220710-231618', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220711-010539', compile=False)
    # unet_model = tf.keras.models.load_model('models/ours_multi_single_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220711-032611', compile=False)
    # unet_model = tf.keras.models.load_model('models/unetpp_efficientnetb2_32,64,128,256,512_cubicasa5k_augment_20220708-105101', compile=False)
    # unet_model = tf.keras.models.load_model('models/unetpp_efficientnetb2_32,64,128,256,512_cubicasa5k_augment_20220711-091029', compile=False)
    # unet_model = tf.keras.models.load_model('models/cubi_unet3p_20220623-013908', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p_EfficientNetB2_64,128,256,512,1024_r3d_augment_20220721-191743', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p_EfficientNetB2_64,128,256,512,1024_r3d_augment_20220721-200822', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p_EfficientNetB2_64,128,256,512,1024_r3d_augment_20220721-223128', compile=False)
    # unet_model = tf.keras.models.load_model('models/ours_multi_EfficientNetB2_64,128,256,512,1024_r3d_augment_20220722-123937', compile=False)

    # unet_model = tf.keras.models.load_model('models/unet3p_EfficientNetB2_64,128,256,512,1024_cubicasa5k_augment_20220722-222000', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p_EfficientNetB2_64,128,256,512,1024_cubicasa5k_augment_20220723-003805', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p_EfficientNetB2_64,128,256,512,1024_cubicasa5k_augment_20220723-024214', compile=False)
    # unet_model = tf.keras.models.load_model('models/ours_multi_EfficientNetB2_64,128,256,512,1024_cubicasa5k_augment_20220723-043436', compile=False)

    # unet_model = tf.keras.models.load_model('models/unet3p_EfficientNetB2_64,128,256,512,1024_cubicasa5k_augment_20220724-105507', compile=False)
    # unet_model = tf.keras.models.load_model('models/zeng_zeng_cubi_final_vgg16_64,128,256,512,1024_cubicasa5k_augment_20220725-194943', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p_EfficientNetB2_64,128,256,512,1024_cubicasa5k_augment_20220724-171411', compile=False)
    # unet_model = tf.keras.models.load_model('models/cab2_map_abc_cubi_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment_20220811-200057', compile=False)

    # unet_model = tf.keras.models.load_model('models/cab2_map_abc_combi_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220812-204123', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p_unet3p_combi_EfficientNetB2_32,64,128,256,512_cubicasa5k_augment,multi_plans_augment_20220813-051541', compile=False)

    # unet_model = tf.keras.models.load_model('models/unet3p_gn_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220731-091658', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p_unet3p1_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220727-224157', compile=False)
    # unet_model = tf.keras.models.load_model('models/zeng_zeng_multi_final_vgg16_32,64,128,256,512_multi_plans_augment_20220728-135338', compile=False)
    # unet_model = tf.keras.models.load_model('models/unet3p_2samples_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220731-153551', compile=False)
    # unet_model = tf.keras.models.load_model('models/cab1_hhdc_cam_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220731-215239', compile=False)
    # unet_model = tf.keras.models.load_model('models/cab1_cam_EfficientNetB2_32,64,128,256,512_multi_plans_augment_20220801-020641', compile=False)

    # unet_model = tf.keras.models.load_model('models/multi_zeng1', compile=False)
    # unet_model = tf.keras.models.load_model('models/multi_cubi1', compile=False)
    # unet_model = tf.keras.models.load_model('models/multi_ours1', compile=False)
    # unet_model = tf.keras.models.load_model('models/multi_unet3p3', compile=False)

    # unet_model = tf.keras.models.load_model('models/combi_zeng1', compile=False)
    # unet_model = tf.keras.models.load_model('models/combi_unet3p1', compile=False)

    for model, normalize in models:
        model_name = model.split('_')[0]
        unet_model = tf.keras.models.load_model('models/' + model, compile=False)
        print("Normalize: {0}".format(normalize))

        tta = False

        predict = False
        if predict:
            # newcubi1 = cv2.imread('resources/newcubi1.png')

            # cubi1 = cv2.imread('resources/cubi1.png')
            # cubi2 = cv2.imread('resources/cubi2.png')
            # cubi3 = cv2.imread('resources/cubi3.png')
            # cubi4 = cv2.imread('resources/cubi4.png')

            single = cv2.imread('resources/single.jpg')
            # multi = cv2.imread('resources/multi.jpg')
            # # image = cv2.imread('resources/multi_large.jpg')
            # # image = cv2.imread('resources/multi_largest.jpg')
            # m_sampled = cv2.imread('resources/m_sampled.jpg')
            # m_sampled2 = cv2.imread('resources/m_sampled2.jpg')
            # mplan_s = cv2.imread('resources/mplan_s.jpg')
            mplan_s = cv2.imread('resources/emc.jpg')
            # mplan_s = cv2.imread('resources/mplan_s_r.png')
            # mplan_s = cv2.imread('resources/multi_plan2.jpg')
            # gv = cv2.imread('resources/gv2.jpg')

            # plt.figure(dpi=400)
            # image_org = cv2.cvtColor(mplan_s, cv2.COLOR_RGB2BGR)
            # image_org, pads = pad_to(image_org, 64)
            # plt.imshow(image_org)
            # # plt.imshow(ind2rgb(rgb2ind(cv2.cvtColor(cv2.imread('resources/pp_original.png'), cv2.COLOR_RGB2BGR))),
            # #            alpha=0.66)
            # plt.imshow(ind2rgb(rgb2ind(cv2.cvtColor(cv2.imread('resources/pp_result_refined.png'), cv2.COLOR_RGB2BGR))),
            #            alpha=0.66)
            # plt.axis('off')
            # # timestr = 'prediction'
            # timestr = 'prediction_pp'
            # plt.savefig('results/' + timestr, bbox_inches='tight', pad_inches=0)
            # plt.show()
            # exit(0)

            # emcsmall = cv2.imread('resources/emcsmall.jpg')

            resizes = [4200]
            # images = [emcsmall]
            images = [single]
            # images = [single, multi, m_sampled2, mplan_s]
            # images = [single, cubi3, cubi4]
            # images = [newcubi1]
            # images = [newcubi1, cubi4]
            images = [mplan_s]
            for i, image_org in enumerate(images):
                image_org = image_org[100:100 + 1840, 100: 100 + 2040]
                # left = 1880
                # right = left + 1700
                # down = 3200
                # up = down + 1000
                # image_org = image_org[left:right, down: up]

                image_g = cv2.cvtColor(image_org, cv2.COLOR_RGB2GRAY)
                image_org = image_org.astype(np.float32)
                l = np.mean(image_org[image_g < 250, :])
                u = np.mean(image_org)
                print(l, u)

                image_ind = np.logical_and(image_g < u, image_g > l)
                image_org[image_ind, :] *= 1 / 2
                image_org = image_org.astype(np.uint8)

                # plt.figure(dpi=200)
                # plt.imshow(image_org)
                # plt.show()
                # exit(0)

                # cv2.imwrite('emc_resize.jpg', image_org)

                # for d1 in [250]:
                #     for d2 in range(180, 245, 5):
                #         image_test = np.copy(image_org)
                #         image_ind = np.logical_and(image_g < d1, image_g > d2)
                #         image_test[image_ind, :] *= 1 / 2
                #         image_test = image_test.astype(np.uint8)
                #         cv2.imwrite('emc_resize_{0}_{1}.jpg'.format(d1, d2), image_test)
                # exit(0)

                resize = True
                if resize:
                    w = resizes[i]
                    if image_org.shape[1] > w:
                        h = int(w / image_org.shape[1] * image_org.shape[0])
                        image_org = cv2.resize(image_org, (w, h), interpolation=cv2.INTER_AREA)
                image_org = np.clip(image_org, 0, 255)

                # cv2.imwrite('emc_contrast_before.jpg', cv2.cvtColor(image_org, cv2.COLOR_RGB2BGR))
                # cv2.imwrite('emc_contrast_after.jpg', cv2.cvtColor(image_org, cv2.COLOR_RGB2BGR))

                # image_org = image_org[100:100 + 1040, 100: 100 + 1040]

                # image_org = cv2.resize(image_org, [512, 512])
                # image_org = cv2.resize(image_org, [640, 640])
                # image_org = cv2.resize(image_org, [832, 832])
                image_org, pads = pad_to(image_org, 32)

                # plot_diff(image_org)

                image = tf.convert_to_tensor(image_org, dtype=tf.uint8)
                image = tf.cast(image, dtype=tf.float32)

                shp = image.shape
                if normalize:
                    image = tf.reshape(image, [shp[0], shp[1], 3]) / 255  # Only divide if not effnet
                else:
                    image = tf.reshape(image, [shp[0], shp[1], 3])

                if tta:
                    image = image.numpy()
                    image_batch = np.array([
                        image,
                        np.fliplr(image),
                        np.flipud(image),
                        np.rot90(image, k=1),
                        np.rot90(image, k=3),
                    ])
                else:
                    image_batch = tf.expand_dims(image, axis=0)

                prediction = unet_model.predict_on_batch(image_batch)
                if isinstance(prediction, tuple):
                    prediction = prediction[0]
                if tta:
                    results = [
                        prediction[0].argmax(axis=-1),
                        np.fliplr(prediction[1].argmax(axis=-1)),
                        np.flipud(prediction[2].argmax(axis=-1)),
                        np.rot90(prediction[3].argmax(axis=-1), k=-1),
                        np.rot90(prediction[4].argmax(axis=-1), k=-3),
                    ]
                    result = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0,
                                                 arr=np.array(results)).astype(
                        np.uint8)
                    # result = np.mean(results, axis=0)
                else:
                    result = prediction[0].argmax(axis=-1).astype(np.uint8)
                # result_pp = post_process(np.copy(result))

                # Unpad result
                # TODO
                timestr = time.strftime("%Y%m%d-%H%M%S")
                overlay = True
                if overlay:
                    plt.figure(dpi=400)
                    plt.imshow(image_org)
                    plt.imshow(ind2rgb(result), alpha=0.7)
                    plt.axis('off')
                    plt.savefig('results/' + timestr, bbox_inches='tight', pad_inches=0)
                    plt.show()
                mpimg.imsave("results/result" + timestr + str(i) + '.png', ind2rgb(result))
                # mpimg.imsave("results/result_pp" + timestr + str(i) + '.png', ind2rgb(result_pp))
        else:
            classes = ['walls', 'openings']
            # classes = ['walls', 'railings', 'doors', 'windows', 'stairs_all']
            # classes = ['walls', 'glass_walls', 'railings', 'doors', 'sliding_doors', 'windows', 'stairs_all']
            classes = ['bg'] + classes
            datasets = [
                # 'multi_plans_test_augment'
                # 'multi_plans_augment'
                # 'cubicasa5k_augment',
                'r3d_augment'
            ]

            test_dataset = floorplans.load_test_data(classes, datasets, normalize=normalize)

            # al = 0
            # ul = 0
            # for (image, label) in test_dataset.take(10).batch(1):
            #     prediction = unet_model.predict_on_batch(image)
            #     av = aaf_test(label, prediction, len(classes))
            #     uv = ul_test(label, prediction)
            #     al += av
            #     ul += uv
            #     print('af', av)
            #     print('ul', uv)
            # print(al, ul)

            i = 0
            for (image, label) in test_dataset.skip(0).take(25).batch(1):
                i += 1
                prediction = unet_model.predict_on_batch(image)
                if isinstance(prediction, tuple):
                    prediction = prediction[1]
                write('input_' + str(i), model_name, np.array(image[0], dtype=np.uint8))

                conversion = None
                if len(datasets) == 1:
                    if datasets[0] == 'r3d_augment':
                        conversion = 'r3d'
                    elif datasets[0] == 'cubicasa5k_augment':
                        conversion = 'cubicasa5k'

                write('label_' + str(i), model_name, ind2rgb(label[0, :, :, :len(classes)].numpy().argmax(axis=-1), conversion=conversion))

                write('result_' + str(i), model_name, ind2rgb(prediction[0].argmax(axis=-1), conversion=conversion))
            # rows = 2
            # skip = 0
            # for t in range(1):
            #     fig, axs = plt.subplots(rows, 3, figsize=(20, rows * 8))
            #     # for ax, (image, label) in zip(axs, test_dataset.shuffle(64).take(rows).batch(1)):
            #     for ax, (image, label) in zip(axs, test_dataset.skip(skip).take(rows).batch(1)):
            #         prediction = unet_model.predict_on_batch(image)
            #         if isinstance(prediction, tuple):
            #             prediction = prediction[1]
            #         print(image[0].shape)
            #         ax[0].matshow(np.array(image[0], dtype=np.uint8))
            #         ax[1].matshow(ind2rgb(label[0, :, :, :len(classes)].numpy().argmax(axis=-1)))
            #         ax[2].matshow(ind2rgb(prediction[0].argmax(axis=-1)))
            #
            #         ax[0].axis('off')
            #         ax[1].axis('off')
            #         ax[2].axis('off')
            #
            #     timestr = time.strftime("%Y%m%d-%H%M%S")
            #     plt.savefig('results/' + timestr)
            #     plt.show()


if __name__ == "__main__":
    tic = time.time()
    main()
    # predict(['mplan_s.jpg'])
    toc = time.time()
    print('total process time = {} seconds'.format((toc - tic)))
