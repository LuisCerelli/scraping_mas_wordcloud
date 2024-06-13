from flask import Flask, request, jsonify, render_template
import pytube
import whisper
import re
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib
import os
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO
import numpy as np

matplotlib.use('Agg')  # Use Agg backend for headless mode

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        youtube_url = data.get('youtubeUrl')
        text_filename = data.get('textFilename')
        stop_words = data.get('stopWords', '').split(',')
        num_words = data.get('numWords')
        title = data.get('title', 'Describiendo a Milei, según desaprobación y aprobación')
        left_label = data.get('leftLabel', 'Aprueban')
        right_label = data.get('rightLabel', 'Desaprueban')

        if not youtube_url or not text_filename:
            return jsonify({"error": "Faltan parámetros obligatorios"}), 400

        if not num_words:
            num_words = 0
        else:
            num_words = int(num_words)

        # Descargar y transcribir el video
        model = whisper.load_model("small")
        YouTubeVideo = pytube.YouTube(youtube_url)
        audio = YouTubeVideo.streams.get_audio_only()
        audio.download(filename='tmp.mp4')
        result = model.transcribe('tmp.mp4')

        # Guardar transcripción en un archivo de texto
        with open(f'{text_filename}.txt', 'w') as f:
            f.write(result["text"])

        # Leer el archivo de texto y generar la nube de palabras
        with open(f'{text_filename}.txt', 'r', encoding='utf-8') as file:
            text = file.read()

        word_count = generate_wordcloud(text, text_filename, stop_words, num_words, title, left_label, right_label)

        return jsonify({"image_filename": text_filename, "word_count": word_count})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": "Error al generar los archivos", "details": str(e)}), 500


def generate_wordcloud(text, filename, stop_words, num_words, title, left_label, right_label):
    try:
        stop_words_set = set(['el', 'la', 'de', 'y', 'a', 'en', 'que', 'es', 'un', 'una', 'por', 'con', 'mis', 'mi', 'los', 'no', 'del', 'me', 'ni','se','su','para','lo','como','al','las','si', 'eso','sin','hubo','les','milay',])
        stop_words_set.update([word.strip() for word in stop_words])

        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text)

        words = text.split()
        filtered_words = [word for word in words if word not in stop_words_set]
        
        word_freq = Counter(filtered_words)
        if num_words > 0:
            word_freq = dict(word_freq.most_common(num_words))

        # Crear una nueva figura para la imagen final
        fig, ax = plt.subplots(figsize=(12, 8))  # Ajustar el tamaño de la figura
        ax.set_title(title, fontsize=20, weight='bold', pad=100, y=0.9)# Centrando y ajustando el título

        # Generar y mostrar la nube de palabras
        ax.imshow(WordCloud(width=800, height=400, background_color='#EBEBF7', colormap='Dark2').generate_from_frequencies(word_freq), interpolation='bilinear')

        

        # Ocultar los ejes
        ax.axis('off')

        # Intentar cargar la imagen
        try:
            # Cargar la imagen desde un archivo local
            image = Image.open('ImagenesApp3/encabezado_A&L_little.png')
            aspect_ratio = image.size[1] / image.size[0]
            new_height = int(800 * aspect_ratio)
            image = image.resize((800, new_height))
            image = np.array(image)

            # Calcular la posición x y y para centrar la imagen en la figura
            fig_width, fig_height = fig.get_size_inches() * fig.dpi
            image_width, image_height = image.shape[1], image.shape[0]
            
            x = (fig_width - image_width) / 2 # Centrar la imagen horizontalmente
            print(f'x es igual a {x}')
            y = (fig_height - image_height) / 2 + 150  # Aumentar este valor para mover la imagen hacia arriba
            print(f'y es igual a {y}')
            # Añadir la imagen al fondo
            fig.figimage(image, xo=x, yo=y, zorder=-1)

        except Exception as e:
            # Imprimir el error y continuar sin la imagen de fondo
            print(f"Error al cargar o añadir la imagen de fondo: {str(e)}")


        # Añadir etiquetas
        plt.figtext(0.2, 0.15, left_label, fontsize=10, ha='center', va='center')
        plt.figtext(0.8, 0.15, right_label, fontsize=10, ha='center', va='center')

        # Guardar la imagen
        plt.savefig(f'static/{filename}.png', bbox_inches='tight')

        return len(word_freq)
    except Exception as e:
        print(f"Error al generar la nube de palabras: {str(e)}")
        raise

if __name__ == '__main__':
    if not os.path.exists('static'):
        os.makedirs('static')
    app.run(debug=True)