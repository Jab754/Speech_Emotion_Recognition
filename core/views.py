from django.shortcuts import render,redirect
from django.http import HttpResponse,JsonResponse
from django.conf import settings
from django.views.decorators.cache import cache_control
import os
import pyaudio
import numpy as np
import warnings
import wave
import pickle
import soundfile
import librosa
import subprocess
import json 

warnings.filterwarnings("ignore")

AVAILABLE_EMOTIONS = {
    "angry",
    "sad",
    "neutral",
    "happy",
}

THRESHOLD = 100
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
resullt = np.random.choice(np.array
(list(AVAILABLE_EMOTIONS)))
RATE = 16000
sample_width = 2
SILENCE = 30


def extract_feature(file_name, **kwargs):

    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))

    return result

# def record_to_file(path):
#     # "Records from the microphone and outputs the resulting data to 'path'"
#     p = pyaudio.PyAudio()
#     stream = p.open(format=FORMAT, channels=1, rate=RATE,
#                    input=True, frames_per_buffer=CHUNK_SIZE)
#     frames = []
#     silence_frames = 0
#     while True:
#         data = stream.read(CHUNK_SIZE)
#         frames.append(data)
#         if max(abs(np.frombuffer(data, dtype=np.int16))) < THRESHOLD:
#             silence_frames += 1
#         else:
#             silence_frames = 0
#         if silence_frames > SILENCE:
#             break

#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     wf = wave.open(path, 'wb')
#     wf.setnchannels(1)
#     wf.setsampwidth(sample_width)
#     wf.setframerate(RATE)
#     wf.writeframes(b''.join(frames))
#     wf.close()

def save_path(path):
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    # wf.writeframes(b''.join(frames))
    wf.close()

@cache_control(no_cache=True, must_revalidate=True)
def main(request):
    np.random.seed(42)
    if request.method == 'POST':
        # Get the uploaded audio file from the form
        if 'audio' in request.FILES:
            audio_file=request.FILES['audio']

            filename = os.path.join(settings.MEDIA_ROOT, 'uploaded.wav')

            with open(filename, 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)

            model = pickle.load(open("./mlModel/mlp_classifier.model", "rb"))

            # save_path(filename)

            features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1,-1)
            resullt = model.predict(features)[0]

            return render(request, 'main.html',{'result':resullt})
        # else:
        #     file = 'recorded_audio.wav'
        #     filename = os.path.join(settings.MEDIA_ROOT, file)
        #     record_to_file(filename)

        #     model = pickle.load(open("./mlModel/mlp_classifier.model", "rb"))

        #     features = extract_feature(filename, mfcc=True, chroma=True, mel=True).reshape(1,-1)

        #     result = model.predict(features)[0]

        #     return render(request, 'main.html',{'result':result})

    else:
        # Render the audio processing form
        return render(request,'main.html', {'cache_bust': np.random.randint(0, 100000)})
        # return HttpResponse('Recording saved to {}'.format(filepath))

        # audio_file = request.FILES['audio_file']

        # # Save the uploaded audio file to disk
        # filename = os.path.join(settings.MEDIA_ROOT, 'uploaded.wav')
        # with open(filename, 'wb+') as destination:
        #     for chunk in audio_file.chunks():
        #         destination.write(chunk)
        
        # # Load the MLP classifier model
        # model = pickle.load(open("./mlModel/mlp_classifier.model", "rb"))
        # # Extract features from the uploaded audio file

        # # record_to_file(filename)
        # features = extract_feature(filename, mfcc=True, chroma=True, mel=True)
        # features=features.reshape(1,-1)

        # # Make a prediction using the MLP classifier model
        # result = model.predict(features)[0]
        # # result = np.array([result]).reshape(1, -1) # Reshape result array to have two dimensions

        # # # Concatenate the result and features arrays
        # # features_and_result = np.hstack((features, result))

        # # Return the predicted result to the user
        # return HttpResponse("Predicted result for the given audio file: {}".format(result))
        # # return render(request, 'main.html',{'result':result})

# Create your views here.
def index(request):
    return render(request,'index.html')

# def record_audio(request):
#     # record audio and save to media root directory
#     filename = 'recorded_audio.wav'
#     filepath = os.path.join(settings.MEDIA_ROOT, filename)
#     record_to_file(filepath)

#     # generate the URL to the recorded audio file
#     file_url = request.build_absolute_uri(settings.MEDIA_URL + filename)

#     # return a JSON response indicating whether the recording was successful and the URL to the recorded audio file
#     return JsonResponse({'success': True, 'file_url': file_url})
