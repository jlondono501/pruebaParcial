#!pip install facenet-pytorch

################################################################################
#                            RECONOCIMIENTO FACIAL                             #
#                                                                              #                                   #
################################################################################
# coding=utf-8

"""  
pip install scipy
#pip install torch (torch-2.2.2)
pip install facenet-pytorch
pip install opencv-python
pip install matplotlib
pip install pillow
pip install warn
pip install typing
pip install logging-utilities
pip install lib-platform
pip install glob2
pip install urllib3
pip install streamlit_webrtc
# pip install tqdm (ya instalado)
"""
# Librerías
# ==============================================================================
import PIL.Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import warnings
import typing
import logging
import os
import platform
import glob
import PIL
import facenet_pytorch
from typing import Union, Dict
from PIL import Image
from facenet_pytorch import MTCNN
from facenet_pytorch import InceptionResnetV1
from urllib.request import urlretrieve
from tqdm import tqdm 
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from PIL import Image
import streamlit as st


################################################################################
#                          Configuración básica de Streamlit                   #
#                                                                              #                                   #
################################################################################

# ocultar advertencias de desuso que no afectan directamente el funcionamiento de la aplicación
import warnings
warnings.filterwarnings("ignore")

# establecer algunas configuraciones predefinidas para la página, como el título de la página, el ícono del logotipo, el estado de carga de la página (si la página se carga automáticamente o si necesita realizar alguna acción para cargar)
st.set_page_config(
    page_title="Reconocimiento de Rostros",
    page_icon = ":face:"
)

# ocultar la parte del código, ya que esto es solo para agregar algún estilo CSS personalizado pero no es parte de la idea principal
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""


logging.basicConfig(
    format = '%(asctime)-5s %(name)-10s %(levelname)-5s %(message)s', 
    level  = logging.WARNING,
)


st.title("Smart Regions Center")
st.write("Somos un equipo apasionado de profesionales dedicados a hacer la diferencia")
st.write("""
         App Web para tomar asistencia para partiipantes en reuniones
         Por ALFREDO DIAZ CLARO
         """
         )



################################################################################
# 1 . Detectar la posición de caras en una imagen empleando un detector MTCNN. #
#                                                                              #                                   #
################################################################################

# ==============================================================================
def detectar_caras(imagen: Union[PIL.Image.Image, np.ndarray],
    detector: facenet_pytorch.models.mtcnn.MTCNN=None,
    keep_all: bool        = True,
    min_face_size: int    = 20,
    thresholds: list      = [0.6, 0.7, 0.7],
    device: str           = None,
    min_confidence: float = 0.5,
    fix_bbox: bool        = True,
    verbose               = False)-> np.ndarray:
    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser `np.ndarray, PIL.Image`. Recibido {type(imagen)}."
        )

    if detector is None:
        logging.info('Iniciando detector MTCC')
        detector = MTCNN(
                        keep_all      = keep_all,
                        min_face_size = min_face_size,
                        thresholds    = thresholds,
                        post_process  = False,
                        device        = device
                   )
        
    # Detección de caras
    # --------------------------------------------------------------------------
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen).astype(np.float32)
        
    bboxes, probs = detector.detect(imagen, landmarks=False)
    
    if bboxes is None:
        bboxes = np.array([])
        probs  = np.array([])
    else:
        # Se descartan caras con una probabilidad estimada inferior a `min_confidence`.
        bboxes = bboxes[probs > min_confidence]
        probs  = probs[probs > min_confidence]
        
    logging.info(f'Número total de caras detectadas: {len(bboxes)}')
    logging.info(f'Número final de caras seleccionadas: {len(bboxes)}')

    # Corregir bounding boxes
    #---------------------------------------------------------------------------
    # Si alguna de las esquinas de la bounding box está fuera de la imagen, se
    # corrigen para que no sobrepase los márgenes.
    if len(bboxes) > 0 and fix_bbox:       
        for i, bbox in enumerate(bboxes):
            if bbox[0] < 0:
                bboxes[i][0] = 0
            if bbox[1] < 0:
                bboxes[i][1] = 0
            if bbox[2] > imagen.shape[1]:
                bboxes[i][2] = imagen.shape[1]
            if bbox[3] > imagen.shape[0]:
                bboxes[i][3] = imagen.shape[0]

    # Información de proceso
    # ----------------------------------------------------------------------
    if verbose:
        print("----------------")
        print("Imagen escaneada")
        print("----------------")
        print(f"Caras detectadas: {len(bboxes)}")
        print(f"Correción bounding boxes: {ix_bbox}")
        print(f"Coordenadas bounding boxes: {bboxes}")
        print(f"Confianza bounding boxes:{probs} ")
        print("")
        
    return bboxes.astype(int)


################################################################################
#               2. EXTRAER CARAS                                               #
#                                                                              #                                   #
################################################################################
                    
def extraer_caras(imagen: Union[PIL.Image.Image, np.ndarray],
                  bboxes: np.ndarray,
                  output_img_size: Union[list, tuple, np.ndarray]=[160, 160]) -> None:

    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser np.ndarray, PIL.Image. Recibido {type(imagen)}."
        )
        
    # Recorte de cara
    # --------------------------------------------------------------------------
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen)
        
    if len(bboxes) > 0:
        caras = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            cara = imagen[y1:y2, x1:x2]
            # Redimensionamiento del recorte
            cara = Image.fromarray(cara)
            cara = cara.resize(tuple(output_img_size))
            cara = np.array(cara)
            caras.append(cara)
            
    caras = np.stack(caras, axis=0)

    return caras


################################################################################
#              3. CALCULAR EMBEDDING                                           #
#                                                                              #                                   #
################################################################################


def calcular_embeddings(img_caras: np.ndarray, encoder=None,
                        device: str=None) -> np.ndarray: 
    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not isinstance(img_caras, np.ndarray):
        raise Exception(
            f"`img_caras` debe ser np.ndarray {type(img_caras)}."
        )
        
    if img_caras.ndim != 4:
        raise Exception(
            f"`img_caras` debe ser np.ndarray con dimensiones [nº caras, ancho, alto, 3]."
            f" Recibido {img_caras.ndim}."
        )
        
    if encoder is None:
        logging.info('Iniciando encoder InceptionResnetV1')
        encoder = InceptionResnetV1(
                        pretrained = 'vggface2',
                        classify   = False,
                        device     = device
                   ).eval()
        
    # Calculo de embedings
    # --------------------------------------------------------------------------
    # El InceptionResnetV1 modelo requiere que las dimensiones de entrada sean
    # [nº caras, 3, ancho, alto]
    caras = np.moveaxis(img_caras, -1, 1)
    caras = caras.astype(np.float32) / 255
    caras = torch.tensor(caras)
    embeddings = encoder.forward(caras).detach().cpu().numpy()
    embeddings = embeddings
    return embeddings


#################################################################################
#             4.   IDENTIFICAR CARAS                                            #
#                                                                               #                                   #
################################################################################

def identificar_caras(embeddings: np.ndarray,
                      dic_referencia: dict,
                      threshold_similaridad: float = 0.6) -> list:
    
    identidades = []
        
    for i in range(embeddings.shape[0]):
        # Se calcula la similitud con cada uno de los perfiles de referencia.
        similitudes = {}
        for key, value in dic_referencia.items():
            similitudes[key] = 1 - cosine(embeddings[i], value)
        
        # Se identifica la persona de mayor similitud.
        identidad = max(similitudes, key=similitudes.get)
        # Si la similitud < threshold_similaridad, se etiqueta como None
        if similitudes[identidad] < threshold_similaridad:
            identidad = None
            
        identidades.append(identidad)
        
    return identidades
 
 

################################################################################
#                  SOLO ES LLAMADO DE pipeline_deteccion_webcam                #
#              5.  Mostrar la imagen original con las boundig boxES  CV2       #
#                                                                              #                                   #
################################################################################

 
def mostrar_bboxes_cv2(imagen: Union[PIL.Image.Image, np.ndarray],
                       bboxes: np.ndarray,
                       identidades: list=None,
                       device: str='window') -> None:
    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser `np.ndarray`, `PIL.Image`. Recibido {type(imagen)}."
        )
        
    if identidades is not None:
        if len(bboxes) != len(identidades):
            raise Exception(
                '`identidades` debe tener el mismo número de elementos que `bboxes`.'
            )
    else:
        identidades = [None] * len(bboxes)

    # Mostrar la imagen y superponer bounding boxes
    # --------------------------------------------------------------------------      
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen).astype(np.float32) / 255
    
    if len(bboxes) > 0:
        
        for i, bbox in enumerate(bboxes):
            
            if identidades[i] is not None:
                cv2.rectangle(
                    img       = imagen,
                    pt1       = (bbox[0], bbox[1]),
                    pt2       = (bbox[2], bbox[3]),
                    color     = (0, 255, 0),
                    thickness = 2
                )
                
                cv2.putText(
                    img       = imagen, 
                    text      = identidades[i], 
                    org       = (bbox[0], bbox[1]-10), 
                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1e-3 * imagen.shape[0],
                    color     = (0,255,0),
                    thickness = 2
                )
            else:
                cv2.rectangle(
                    img       = imagen,
                    pt1       = (bbox[0], bbox[1]),
                    pt2       = (bbox[2], bbox[3]),
                    color     = (255, 0, 0),
                    thickness = 2
                )
        
    if device is None:
        return imagen
    else:
        # Convertir la imagen de BGR a RGB
        #frame_rgb = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

        # Convertir la imagen a formato PIL
        img_pil = Image.fromarray(imagen)

    return img_pil
        



################################################################################
#            SOLO ES LLAMADO DESDE pipeline_deteccion_imagen                   #
#            5. Mostrar la imagen original con las boundig box                 #
#                                                                              #                                   #
################################################################################


def mostrar_bboxes(imagen: Union[PIL.Image.Image, np.ndarray],
                bboxes: np.ndarray,
                identidades: list=None,
                ax=None,
                device: str='window' ) -> None:

    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not isinstance(imagen, (np.ndarray, PIL.Image.Image)):
        raise Exception(
            f"`imagen` debe ser `np.ndarray, PIL.Image`. Recibido {type(imagen)}."
        )
        
    if identidades is not None:
        if len(bboxes) != len(identidades):
            raise Exception(
                '`identidades` debe tener el mismo número de elementos que `bboxes`.'
            )
    else:
        identidades = [None] * len(bboxes)

    # Mostrar la imagen y superponer bounding boxes
    # --------------------------------------------------------------------------
    if ax is None:
        ax = plt.gca()
        
    if isinstance(imagen, PIL.Image.Image):
        imagen = np.array(imagen).astype(np.float32) / 255
    
    if len(bboxes) > 0:
        
        for i, bbox in enumerate(bboxes):
            
            if identidades[i] is not None:
                cv2.rectangle(
                    img       = imagen,
                    pt1       = (bbox[0], bbox[1]),
                    pt2       = (bbox[2], bbox[3]),
                    color     = (0, 255, 0),
                    thickness = 2
                )
                
                cv2.putText(
                    img       = imagen, 
                    text      = identidades[i], 
                    org       = (bbox[0], bbox[1]-10), 
                    fontFace  = cv2.FONT_HERSHEY_SIMPLEX, 
                    fontScale = 1e-3 * imagen.shape[0],
                    color     = (0,255,0),
                    thickness = 2
                )
            else:
                cv2.rectangle(
                    img       = imagen,
                    pt1       = (bbox[0], bbox[1]),
                    pt2       = (bbox[2], bbox[3]),
                    color     = (255, 0, 0),
                    thickness = 2
                )
        
    if device is None:
        return imagen
    else:
        # Convertir la imagen de BGR a RGB
        #frame_rgb = cv2.cvtColor(imagen, cv2.COLOR_RGB2BGR)

        # Convertir la imagen a formato PIL
        img_pil = Image.fromarray(imagen)

    return img_pil
                
                
################################################################################
#                            MODULO DE CARGA DE FOTOS DE PERSONAS              #
#                            detector MTCC Y encoder InceptionResnetV1         #                                   #
################################################################################
@st.cache_data
def crear_diccionario_referencias(folder_path:str,
                                  dic_referencia:dict=None,
                                  detector: facenet_pytorch.models.mtcnn.MTCNN=None,
                                  min_face_size: int=40,
                                  thresholds: list=[0.6, 0.7, 0.7],
                                  min_confidence: float=0.9,
                                  encoder=None,
                                  device: str=None,
                                  verbose: bool=False)-> dict:
    
    # Comprobaciones iniciales
    # --------------------------------------------------------------------------
    if not os.path.isdir(folder_path):
        raise Exception(
            f"Directorio {folder_path} no existe."
        )
        
    if len(os.listdir(folder_path) ) == 0:
        raise Exception(
            f"Directorio {folder_path} está vacío."
        )
    
    
    if detector is None:
        logging.info('Iniciando detector MTCC')
        detector = MTCNN(
                        keep_all      = False,
                        post_process  = False,
                        min_face_size = min_face_size,
                        thresholds    = thresholds,
                        device        = device
                   )
    
    if encoder is None:
        logging.info('Iniciando encoder InceptionResnetV1')
        encoder = InceptionResnetV1(
                        pretrained = 'vggface2',
                        classify   = False,
                        device     = device
                   ).eval()
        
    
    new_dic_referencia = {}
    folders = glob.glob(folder_path + "/*")
    
    for folder in folders:
        
        if platform.system() in ['Linux', 'Darwin']:
            identidad = folder.split("/")[-1]
        else:
            identidad = folder.split("\\")[-1]
                                     
        logging.info(f'Obteniendo embeddings de: {identidad}')
        embeddings = []
        # Se lista todas las imagenes .jpg .jpeg .tif .png
        path_imagenes = glob.glob(folder + "/*.jpg")
        path_imagenes.extend(glob.glob(folder + "/*.jpeg"))
        path_imagenes.extend(glob.glob(folder + "/*.tif"))
        path_imagenes.extend(glob.glob(folder + "/*.png"))
        logging.info(f'Total imagenes referencia: {len(path_imagenes)}')
        
        for path_imagen in path_imagenes:
            logging.info(f'Leyendo imagen: {path_imagen}')
            imagen = Image.open(path_imagen)
            # Si la imagen es RGBA se pasa a RGB
            if np.array(imagen).shape[2] == 4:
                imagen  = np.array(imagen)[:, :, :3]
                imagen  = Image.fromarray(imagen)
                
            bbox = detectar_caras(
                        imagen,
                        detector       = detector,
                        min_confidence = min_confidence,
                        verbose        = False
                    )
            
            if len(bbox) > 1:
                logging.warning(
                    f'Más de 2 caras detectadas en la imagen: {path_imagen}. '
                    f'Se descarta la imagen del diccionario de referencia.'
                )
                continue
                
            if len(bbox) == 0:
                logging.warning(
                    f'No se han detectado caras en la imagen: {path_imagen}.'
                )
                continue
                
            cara = extraer_caras(imagen, bbox)
            embedding = calcular_embeddings(cara, encoder=encoder)
            embeddings.append(embedding.flatten())
        
        if verbose:
            print(f"Identidad: {identidad} --- Imágenes referencia: {len(embeddings)}")
            
        embedding_promedio = np.array(embeddings).mean(axis = 0)
        new_dic_referencia[identidad] = embedding_promedio
        
    if dic_referencia is not None:
        dic_referencia.update(new_dic_referencia)
        return dic_referencia
    else:
        return new_dic_referencia
    

def pipeline_deteccion_imagen(imagen: Union[PIL.Image.Image, np.ndarray],
                              dic_referencia:dict,
                              detector: facenet_pytorch.models.mtcnn.MTCNN=None,
                              keep_all: bool=True,
                              min_face_size: int=20,
                              thresholds: list=[0.6, 0.7, 0.7],
                              device: str=None,
                              min_confidence: float=0.5,
                              fix_bbox: bool=True,
                              output_img_size: Union[list, tuple, np.ndarray]=[160, 160],
                              encoder=None,
                              threshold_similaridad: float=0.5,
                              ax=None,
                              verbose=False)-> None:
    

    bboxes = detectar_caras(
                imagen         = imagen,
                detector       = detector,
                keep_all       = keep_all,
                min_face_size  = min_face_size,
                thresholds     = thresholds,
                device         = device,
                min_confidence = min_confidence,
                fix_bbox       = fix_bbox
              )
    
    if len(bboxes) == 0:
        
        logging.info('No se han detectado caras en la imagen.')
        mostrar_bboxes(
            imagen      = imagen,
            bboxes      = bboxes,
            ax          = ax
        )
        
    else:
    
        caras = extraer_caras(
                    imagen = imagen,
                    bboxes = bboxes
                )

        embeddings = calcular_embeddings(
                        img_caras = caras,
                        encoder   = encoder
                     )

        identidades = identificar_caras(
                         embeddings     = embeddings,
                         dic_referencia = dic_referencias,
                         threshold_similaridad = threshold_similaridad
                       )

        frame_procesado= mostrar_bboxes(
            imagen      = imagen,
            bboxes      = bboxes,
            identidades = identidades,
            ax          = ax
            )
        
    return imagen ,identidades
    



################################################################################
#                            MODULO DE DETECCION WEBCAMM                       #
#                                                                              #                                   #
################################################################################


def pipeline_deteccion_webcam(frame,dic_referencia, output_device='window', detector=None, keep_all=True, min_face_size=40, thresholds=[0.6, 0.7, 0.7], device=None, min_confidence=0.5, fix_bbox=True, output_img_size=[160, 160], encoder=None, threshold_similaridad=0.5, ax=None, verbose=False):
    # Función para capturar una sola foto desde la webcam y procesarla


    #image = Image.open(frame)
    #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    bboxes = detectar_caras(
                    imagen         = frame,
                    detector       = detector,
                    keep_all       = keep_all,
                    min_face_size  = min_face_size,
                    thresholds     = thresholds,
                    device         = device,
                    min_confidence = min_confidence,
                    fix_bbox       = fix_bbox
                )

    if len(bboxes) == 0:
        logging.info('No se han detectado caras en la imagen. paso detectar')
        # Convertir la imagen de BGR a RGB
        print('no hat boxes')          
        return None
    
    caras = extraer_caras(
                imagen = frame,
                bboxes = bboxes
            )
    
    embeddings = calcular_embeddings(
                    img_caras = caras,
                    encoder   = encoder
                )

    identidades = identificar_caras(
                    embeddings     = embeddings,
                    dic_referencia = dic_referencia,
                    threshold_similaridad = threshold_similaridad
                )

    frame_procesado = mostrar_bboxes_cv2(
                        imagen      = frame,
                        bboxes      = bboxes,
                        identidades = identidades,
                        device = output_device
                    )

    return frame_procesado ,identidades


# ==============================================================================
#
#
#                      I N I C I O    D E L  A P P 
#
#
# ==============================================================================
# Detectar si se dispone de GPU cuda
# ==============================================================================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(F'Running on device: {device}')
 
 # Crear diccionario de referencia para cada persona
# ==============================================================================
dic_referencias = crear_diccionario_referencias(
                    folder_path    = './images',
                    min_face_size  = 40,
                    min_confidence = 0.9,
                    device         = device,
                    verbose        = True
                  )

 
 

################################################################################
#                            MODULO DE STREAMLIT                               #
#                                                                              #                                   #
################################################################################

st.title("Tomar Asistencia en Reunión")
st.text('Requiere que los asistente se hayan inscrito con anterioridad')

tab1, tab2 = st.tabs(["Tomar Foto", "Cargar Foto"])

with tab1:
    st.subheader("Tomar Foto con la Cámara")
    frame = st.camera_input("Tome una foto") 
    if frame is None:
        st.text("Por favor tome una foto")   
    else:
        imagen_pil = PIL.Image.open(frame)
        #Convertir la imagen PIL a un array numpy
        imagen_np = np.array(imagen_pil)
        
        imagen,identidades=pipeline_deteccion_webcam(imagen_np ,dic_referencias, 0.4)
        if imagen is None:
            st.write("No se ha cargado ni tomado ninguna foto.")
        else:
            print(type(imagen))
       
            st.image(imagen , caption='La Foto Tomada tiene las siguientes personas')
            st.write("Las personas en la reunión son: ",identidades)

with tab2:
    st.subheader("Cargar Foto desde el Computador")
    
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
    
if uploaded_file is not None:
    # Convertir el archivo cargado a un objeto de imagen PIL
    imagen_pil = PIL.Image.open(uploaded_file)
    
    # Convertir la imagen PIL a un array numpy
    imagen_np = np.array(imagen_pil)
    
    # Llamar a la función pipeline_deteccion_imagen con la imagen procesada
    imagen,identidades=pipeline_deteccion_imagen(
        imagen                = imagen_np,
        dic_referencia        = dic_referencias,
        min_face_size         = 20,
        thresholds            = [0.6, 0.7, 0.7],
        min_confidence        = 0.5,
        threshold_similaridad = 0.6,
        device                = device,
        ax                    = None,  # No estás usando 'ax' en este caso
        verbose               = False
    )
    
    if imagen is None:
        st.write("No se ha cargado ni tomado ninguna foto.")
    else:
        st.image(imagen, caption='La Foto subida tiene las siguientes personas')
        st.write("Las personas en la reunión son: ",identidades)