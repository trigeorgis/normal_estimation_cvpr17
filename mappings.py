from functools import lru_cache
import menpo.io as mio
import menpo3d.io as m3io
import numpy as np


# load the maps between LSFM/Basel etc
@lru_cache()
def map_tddfa_to_basel():
    maps = mio.import_pickle(
        '/vol/atlas/databases/itwmm/mapping_mein3d_to_tddfa.pkl.gz')
    return maps['map_tddfa_to_basel']


def fw_to_fw_cropped():
    return mio.import_pickle('/vol/atlas/databases/itwmm/3ddfa_to_trimmed_no_neck_mask.pkl.gz')


@lru_cache()
def template_fw():
    return mio.import_pickle('/vol/atlas/databases/itwmm/mein3d_fw_correspond_mean.pkl.gz')


def template_fw_cropped():
    return template_fw().from_mask(fw_to_fw_cropped())


# Remappings between BFM [] - Face warehouse [fw] - Face warehouse cropped [fwc]
def map_basel_shape_model_to_fw(shape_model):
    shape_model = shape_model.copy()
    c  = shape_model._components.reshape([shape_model._components.shape[0], -1, 3])
    shape_model._components = c[:, map_tddfa_to_basel()].reshape([shape_model._components.shape[0], -1])
    shape_model._mean = shape_model._mean.reshape([-1, 3])[map_tddfa_to_basel()].ravel()
    shape_model.template_instance = template_fw().from_vector(shape_model._mean)
    return shape_model


def map_basel_shape_model_to_fwc(shape_model):
    shape_model = shape_model.copy()
    c  = shape_model._components.reshape([shape_model._components.shape[0], -1, 3])
    shape_model._components = c[:, map_tddfa_to_basel()][:, fw_to_fw_cropped()].reshape([shape_model._components.shape[0], -1])
    shape_model._mean = shape_model._mean.reshape([-1, 3])[map_tddfa_to_basel()][fw_to_fw_cropped()].ravel()
    shape_model.template_instance = template_fw_cropped().from_vector(shape_model._mean)
    return shape_model


def map_basel_texture_model_to_fw(texture_model):
    texture_model = texture_model.copy()
    c  = texture_model._components.reshape([texture_model._components.shape[0], -1, 3])
    texture_model._components = c[:, map_tddfa_to_basel()].reshape([texture_model._components.shape[0], -1])
    texture_model._mean = texture_model._mean.reshape([-1, 3])[map_tddfa_to_basel()].ravel()
    return texture_model


def map_basel_texture_model_to_fwc(texture_model):
    texture_model = texture_model.copy()
    c  = texture_model._components.reshape([texture_model._components.shape[0], -1, 3])
    texture_model._components = c[:, map_tddfa_to_basel()][:, fw_to_fw_cropped()].reshape([texture_model._components.shape[0], -1])
    texture_model._mean = texture_model._mean.reshape([-1, 3])[map_tddfa_to_basel()][fw_to_fw_cropped()].ravel()
    return texture_model


# Remap basel landmarks to fw landmarks by expressing as fw indices
def fw_index_for_lms():
    basel_model, landmarks = load_basel_shape()
    basel_mean = basel_model.mean()
    basel_index = np.argmin(basel_mean.distance_to(landmarks), axis=0)

    m = np.ones(basel_mean.n_points) * -1
    m[basel_index] = np.arange(68)

    poses = np.where((m[map_tddfa_to_basel()] >= 0))[0]

    new_ids = m[map_tddfa_to_basel()][poses]
    return poses[np.argsort(new_ids)]


def load_basel_shape():
    shape_model = mio.import_pickle('/vol/atlas/databases/lsfm/shape_PCAModel.pkl', encoding='latin1')
    landmarks = m3io.import_landmark_file('./template.ljson').lms
    return shape_model, landmarks


def load_basel_texture():
    return mio.import_pickle('./texture_PCAModel.pkl', encoding='latin1')


def load_basel_shape_fw():
    shape_model, landmarks = load_basel_shape()
    return map_basel_shape_model_to_fw(shape_model), landmarks


def load_basel_shape_fwc():
    shape_model, landmarks = load_basel_shape()
    return map_basel_shape_model_to_fwc(shape_model), landmarks


def load_basel_texture_fw():
    return map_basel_texture_model_to_fw(load_basel_texture())


def load_basel_texture_fwc():
    return map_basel_texture_model_to_fwc(load_basel_texture())


def load_lsfm_shape_fwc():
    tr = mio.import_pickle('/vol/atlas/databases/lsfm/corrective_translation.pkl')
    shape_model = mio.import_pickle('/vol/atlas/databases/lsfm/lsfm_shape_model_fw_cropped.pkl')
    landmarks = tr.apply(m3io.import_landmark_file('template.ljson').lms)
    return shape_model, landmarks


def load_lsfm_texture_fwc():
    return mio.import_pickle('/vol/atlas/databases/lsfm/colour_pca_model_fw_cropped.pkl')


def load_lsfm_combined_fw():
    shape_model = mio.import_pickle('/vol/atlas/databases/lsfm/combined_model.pkl')
    landmarks = m3io.import_landmark_file('template.ljson').lms.from_vector(shape_model.mean().points[fw_index_for_lms()])
    return shape_model, landmarks


def load_basel_combined_fw():
    shape_model = mio.import_pickle('/vol/atlas/databases/lsfm/basel_combined_model_fw.pkl')
    landmarks = m3io.import_landmark_file('template.ljson').lms.from_vector(shape_model.mean().points[fw_index_for_lms()])
    return shape_model, landmarks


def load_itwmm_texture_rgb_fwc():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw_cropped/rgb/rgb_per_vertex_fw_cropped_texture_model.pkl')


def load_itwmm_texture_fast_dsift_fwc():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw_cropped/fast_dsift/pca_model.pkl')


def load_itwmm_texture_fast_dsift_fw():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw/fast_dsift.pkl')


def load_itwmm_texture_rgb_fw():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw/rgb.pkl')


def load_itwmm_texture_no_mask_fast_dsift_fw():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw_no_mask/fast_dsift.pkl')


def load_itwmm_texture_no_mask_rgb_fw():
    return mio.import_pickle('/vol/atlas/databases/itwmm/itwmm_texture/per_vertex_fw_no_mask/rgb.pkl')


def load_fw_mean_id_expression_fw():
    shape_model = mio.import_pickle('/vol/atlas/databases/lsfm/expression_model_id_mean.pkl')
    landmarks = m3io.import_landmark_file('template.ljson').lms.from_vector(shape_model.mean().points[fw_index_for_lms()])
    return shape_model, landmarks


def load_fw_expression_fwc():
    expression_model = mio.import_pickle('./identity_texture_emotion.pkl')['expression']
    expression_model._components /= 100000
    expression_model._mean /= 100000
    tr = mio.import_pickle('/vol/atlas/databases/lsfm/corrective_translation.pkl')
    expression_model._components = tr.apply(expression_model._components.reshape(29, -1, 3)).reshape(29, -1)
    expression_model._mean = tr.apply(expression_model._mean.reshape(-1, 3)).reshape(-1)
    expression_model.n_active_components = 5
    return expression_model
