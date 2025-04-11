"""
Este script está diseñado para que las personas puedan obtener audio/video a texto 
Se pueden usar 2 estrategias
Una pagada y dos  gratuita con modelos descargables o usando HuggingFace
En este script se verán ambas opciones
""" 

#### Iris Startup Lab
#### Fernando Dorantes Nieto
import pandas as pd
import numpy as np 
import moviepy
import whisper
import whisperx
import time 
import assemblyai as aai
from pydub import AudioSegment


videoFileClip = moviepy.VideoFileClip

#### Estas funciones son generales. Sirven para convertir video a audio
### Se pueden usar si se necesitan 

def videoToAudio(video, audio_path_file):
    video_clip = videoFileClip(video)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path_file, codec='pcm_s16le')

def convert_audio(file, new_file, newFormat):
    import os 
    from pydub import AudioSegment
    #import re 
    fileName, fileExt = os.path.splitext(file)
    fileExt = fileExt.replace('.', '') 
    mainSound = AudioSegment.from_file(file, format = fileExt)
    mainSound.export(new_file, format= newFormat)

#### A continuación se muestran la configuración y las  funciones de Assembly AI el cual es de paga
assemblyApiKey =  'key de assembly AI'

aai.settings.api_key = assemblyApiKey

transcriber = aai.Transcriber()


config = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=2,
            language_code="es") ### Se selecciona el lenguaje de la transcripción del texto

def transcriptUsingAssembly(audio):
    transcript = transcriber.transcribe(audio, config=config)
    speaker_transcripts = []
    for utterance in transcript.utterances:
        speaker_transcripts.append({
            "speaker": f"Speaker {utterance.speaker}",
            "text": utterance.text
        })
    return pd.DataFrame(speaker_transcripts)

def speechToDfAssembly(folder_audios, list_audios):
    import time 
    dfs = []
    i = 0
    audiostoConvert = len(list_audios) - i
    for audio in list_audios:
        i = i +1 
        print(audio)
        soundFile = folder_audios  +  '/'  +  audio
        textResults = transcriptUsingAssembly(soundFile)
        textResults['audio_name'] = audio
        dfs.append(textResults)
        time.sleep(3)
    return dfs





#### Estas variables son para la configuración de Whisper, WhisperX y Hugging Face
device = 'cpu' ### Puede cambiarse a GPU o TPU si tienes estos recursos
batch_size = 16
compute_type ='int8'
hfToken = 'el token de hugging face para la diferenciación de hablante'

whisperModel = whisper.load_model('medium')
whisperModelX = whisperx.load_model('small', 
                                device = device,
                                compute_type= compute_type)


diarizeModel = whisperx.DiarizationPipeline(use_auth_token = hfToken, 
                                            device = device)

#### Usando Whisper simple si es que no se necesita detectar el speaker 
def SpeechToTextWhisperSimple(soundFile):
    import pandas as pd 
    textResults = whisperModel.transcribe(soundFile)
    mainDf = pd.DataFrame(textResults['segments'])
    return mainDf



def speechToDfWhisperSimple(folder_audios, list_audios):
    dfs = []
    i = 0
    audiostoConvert = len(list_audios) - i
    for audio in list_audios:
        i = i +1 
        soundFile = folder_audios  +  '/'  +  audio
        textResults = SpeechToTextWhisperSimple(soundFile)
        textResults['audio_name'] = audio
        textResults = textResults[['audio_name', 'text', 'temperature']]
        dfs.append(textResults)
        print(f'Audios to convert {audiostoConvert}')
        print(i)
        time.sleep(3)
    return dfs


#### Usando Whisper X para detectar al hablante
def SpeechToTextWhisperX(soundFile):
    import pandas as pd 
    soundFile = whisperx.load_audio(soundFile)
    transcription = whisperModelX.transcribe(audio = soundFile, 
                                           batch_size=batch_size)
    model, metadata = whisperx.load_align_model(transcription.get('language', 'es'), 
                                                 device=device) 
    
    transcription = whisperx.align(transcription['segments'],
                                   model, 
                                   metadata, 
                                   soundFile, 
                                   device, 
                                   return_char_alignments=False)
    diarizeSegments= diarizeModel(soundFile)
    mainResult = whisperx.assign_word_speakers(diarizeSegments, transcription)
    mainDf = pd.DataFrame(mainResult['segments'])[['start', 'end', 'text', 'speaker']]
    return mainDf
