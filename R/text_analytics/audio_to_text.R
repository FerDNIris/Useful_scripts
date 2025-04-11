### Este script es para poder analizar audios con whisper en R
### No es un código tan eficiente como en Python 
#remotes::install_github("bnosac/audio.whisper", ref = "0.3.3")
library(audio.whisper)
library(dplyr)
library(av)

#sessionInfo()

speechToTextWhisper <- function(soundFile){
    model <- whisper('locación del modelo') 
    ### Encontrado en la página oficial de Whisper https://github.com/openai/whisper
    trascription <- predict(model, newdata = soundFile, language = "es")
    return(transcription)
}


