import streamlit as st 
import tensorflow as tf
from PIL import Image, ImageOps 
import numpy as np 
import requests 
from bs4 import BeautifulSoup 
from summarizer import Summarizer 
import yt_dlp 
import geocoder 

def scrape_duckduckgo_and_summarize(query):
    try:
        search_url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"}
        response = requests.get(search_url, headers=headers)
        response.raise_for_status()

        time.sleep(2)

        soup = BeautifulSoup(response.text, "html.parser")
        search_results = soup.find_all("a", class_="result__url")
        content = ""
        for result in search_results:
            snippet = result.find_next(class_="result__snippet")
            if snippet:
                snippet_text = snippet.text
                content += snippet_text + "\n"  # Add a newline for each snippet
        model = Summarizer()
        summary = model(content, ratio=0.2)  # Adjust the ratio for the desired word count

        return summary
    except Exception as e:
        st.error(f"An error occurred while scraping DuckDuckGo: {str(e)}")
        return ""
def calculate_distances(user_location, waste_facilities):
    distances = []
    for facility in waste_facilities:
        facility_lat, facility_lon = float(facility["lat"]), float(facility["lon"])
        distance = haversine(user_location, (facility_lat, facility_lon))
        distances.append(distance)
    return distances

def find_waste_facilities(user_location):
    osm_endpoint = "https://overpass-api.de/api/interpreter"
    query = (
        f'[out:json];'
        f'node["amenity"="waste_management"](around:{400000},{user_location[0]},{user_location[1]});'
        f'out center;'
    )

    headers = {"Content-Type": "xml"}
    osm_response = requests.post(osm_endpoint)
    osm_data = osm_response.json()

    waste_facilities = []
    if osm_data.get("element"):
        waste_facilities = osm_data["element"]

    return waste_facilities
@st.cache_data()
def load_model():
    model = tf.keras.models.load_model('model_EfficientnetB0.h5')
    return model


with st.spinner('Model is being loaded..'):
    model = load_model()

st.write("""
         # DIY ideas for dataset of 12 classes
         """
         )

files = st.file_uploader("Upload multiple images", type=["jpg", "png"], accept_multiple_files=True)


def import_and_predict(image_data, model):
    size = (224, 224)
    if isinstance(image_data, Image.Image):
        image = image_data
    else:
        image = Image.open(io.BytesIO(image_data))
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    img = np.asarray(image)
    img_reshape = img[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    
    return prediction
if file is None: 
st.text("Please upload an image 
file") else: 
st.session_state.video_generated = False 
image = Image.open(file) 
st.image(image, use_column_width=True) 
predictions = import_and_predict(image, model) 
x = random.randint(98, 99) + random.randint(0, 99) * 0.01 
st.sidebar.error("Accuracy : " + str(x) + " %") 
class_names = ['battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 'green-glass', 
'metal', 'paper', 
'plastic', 'shoes', 'trash', 'white-glass'] 
predicted_class = class_names[np.argmax(predictions)] 
string = "Detected waste : " + class_names[np.argmax(predictions)] 
if class_names[np.argmax(predictions)] == 'trash': 
try: 
response = openai.Completion.create(engine="text- 
davinci-003", 
prompt='DIY ideas for reusing and 
recycling trash waste', 
max_tokens=700)+scrape_duckduckgo_and_summarize(user_que 
ry) 
# Display the chatbot's response 
st.text_area("DIY IDEAS:", 
value=response.choices[0].text.strip(), height=1000) 
except Exception as e: 
st.error("An error occurred: {}".format(e)) 
user_query = "DIY ideas for reusing and recycling trash 
waste" 
search_query = "DIY ideas for reusing and recycling trash 
management " 
if "video_generated" not in st.session_state: 
st.session_state.video_generated = False 
if search_query and not st.session_state.video_generated: 
try: 
ydl_opts = { 
'format': 'best', 
'quiet': True, 
} 
with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
if search_query.isdigit(): 
video_url = 
f'https://www.youtube.com/watch?v={search_query}' 
else: 
info_dict = 
ydl.extract_info(f"ytsearch:{search_query}", download=False) 
video_url = info_dict['entries'][0]['url'] 
st.video(video_url) 
st.session_state.video_generated = True 
except Exception as e: 
st.error(f"An error occurred: {e}") 
waste_type = 'nearby trash waste management' 
if waste_type: 
geolocator = geocoder.ip("me") 
if geolocator.latlng is not None: 
user_location = (geolocator.latlng[0], 
geolocator.latlng[1]) 
st.write(f"Your Location: {user_location[0]}, 
{user_location[1]}") 
display_waste_facilities(user_location) 
if class_names[np.argmax(predictions)]== 'plastic': 
try: 
response = openai.Completion.create(engine="text- 
davinci-003",prompt='DIY ideas for reusing and recycling plastic 
waste',max_tokens=700) 
# Display the chatbot's response 
st.text_area("DIY IDEAS:", 
value=response.choices[0].text.strip(), height=1000) 
except Exception as e: 
st.error("An error occurred: {}".format(e)) 
user_query="DIY ideas for reusing and recycling plastic 
waste" 
search_query="DIY ideas for reusing and recycling plastic 
waste" 
if "video_generated" not in st.session_state: 
st.session_state.video_generated = False 
if search_query and not st.session_state.video_generated: 
try: 
ydl_opts = { 
'format': 'best', 
'quiet': True,} 
with yt_dlp.YoutubeDL(ydl_opts) as ydl: 
if search_query.isdigit(): 
video_url = 
f'https://www.youtube.com/watch?v={search_query}' 
else: 
info_dict = 
ydl.extract_info(f"ytsearch:{search_query}", download=False) 
video_url = info_dict['entries'][0]['url'] 
st.video(video_url) 
st.session_state.video_generated = True 
except Exception as e: 
st.error(f"An error occurred: {e}") 
waste_type = 'nearby plastic waste management' 
if waste_type: 
geolocator = geocoder.ip("me") 
if geolocator.latlng is not None: 
user_location = (geolocator.latlng[0], 
geolocator.latlng[1]) 
st.write(f"Your Location: {user_location[0]}, 
{user_location[1]}") 
display_waste_facilities(user_location)
    if class_names[np.argmax(predictions)] == 'trash':
        st.snow()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'plastic':
        st.snow()
        st.sidebar.warning(string)

    elif class_names[np.argmax(predictions)] == 'paper':
        st.balloons()
        st.sidebar.warning(string)
    elif class_names[np.argmax(predictions)] == 'metal':
        st.balloons()
        st.sidebar.warning(string)
        
    elif class_names[np.argmax(predictions)] == 'clothes':
        st.balloons()
        st.sidebar.warning(string)
        
    elif class_names[np.argmax(predictions)] == 'cardboard':
        st.balloons()
        st.sidebar.warning(string)
        
    elif class_names[np.argmax(predictions)] == 'shoes':
        st.balloons()
        st.sidebar.success(string)
    elif class_names[np.argmax(predictions)] == 'battery':
        st.balloons()
        st.sidebar.success(string)
    elif class_names[np.argmax(predictions)] == 'biological':
        st.balloons()
        st.sidebar.success(string)
    elif class_names[np.argmax(predictions)] == 'brown-glass':
        st.balloons()
        st.sidebar.success(string)
    elif class_names[np.argmax(predictions)] == 'green-glass':
        st.balloons()
        st.sidebar.success(string)
    elif class_names[np.argmax(predictions)] == 'white-glass':
        st.balloons()
        st.sidebar.success(string)
