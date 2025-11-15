import streamlit as st
import cv2
import numpy as np
import pickle
from skimage import feature, color
from skimage.filters import gabor
import os
from PIL import Image
import io

# Set page configuration
st.set_page_config(
    page_title="Medicinal Plant Classifier",
    page_icon="🌿",
    layout="wide",
)

# Title and description
st.title("Medicinal Plant Classification")
st.markdown("Upload an image of a medicinal plant leaf to identify its species and get recommendations.")

# Plant information dictionary with recommendations
plant_info = {
    
    "Alpinia Galanga (Rasna)": {
        "common_names": ["Green Amaranth", "Slender Amaranth"],
        "description": "Amaranthus viridis is a cosmopolitan species in the botanical family Amaranthaceae. It is an annual herb with an upright, light green stem that grows to about 60–80 cm in height.",
        "medicinal_uses": [
            "Treatment of inflammation",
            "Digestive aid for constipation",
            "Helpful for respiratory disorders",
            "Treatment for anemia due to high iron content"
        ],
        "preparation": "Leaves can be consumed as vegetable or made into a decoction. For medicinal purposes, a tea can be prepared from the dried leaves.",
        "cautions": "Generally considered safe, but should be avoided by people with kidney problems due to its high oxalate content."
    },
    "Azadirachta Indica (Neem)": {
        "common_names": ["Neem", "Indian Lilac"],
        "description": "Azadirachta indica is an evergreen tree native to the Indian subcontinent. It can grow up to 15-20 meters tall with a semi-straight trunk.",
        "medicinal_uses": [
            "Antibacterial and antifungal properties",
            "Treatment for skin diseases",
            "Blood purifier",
            "Natural insecticide",
            "Dental care for gum disease"
        ],
        "preparation": "Leaves can be made into a paste for skin application. Neem oil is extracted from seeds. Bark and leaves can be dried and powdered for consumption.",
        "cautions": "Not recommended during pregnancy. Can interfere with certain medications."
    },
    "Basella Alba (Basale)": {
        "common_names": ["Malabar Spinach", "Indian Spinach"],
        "description": "Basella alba is a fast-growing, soft-stemmed vine with thick, semi-succulent, heart-shaped leaves. It is commonly found in tropical Asia.",
        "medicinal_uses": [
            "Treatment for constipation",
            "Anti-inflammatory properties",
            "Promotes healthy skin and hair",
            "Supports bone health due to high calcium content"
        ],
        "preparation": "Leaves can be consumed as a vegetable. Juice from leaves can be applied topically for skin conditions.",
        "cautions": "Generally safe for consumption, but should be eaten in moderation due to oxalate content."
    },
    "Piper Betle (Betel)": {
        "common_names": ["Betel Leaf", "Paan"],
        "description": "Betel is a vine belonging to the Piperaceae family. It produces heart-shaped leaves that are commonly used in South and Southeast Asian cultures.",
        "medicinal_uses": [
            "Digestive aid",
            "Stimulant properties",
            "Antiseptic qualities",
            "Relief from respiratory conditions"
        ],
        "preparation": "Leaves are typically consumed fresh, often wrapped around areca nut (though this combination has health risks). For medicinal purposes, juice from leaves can be consumed.",
        "cautions": "Betel quid (with areca nut) is carcinogenic. Excessive consumption may lead to oral health issues."
    },
    "Murraya Koenigii (Curry)": {
        "common_names": ["Sweet Neem", "Karivepaku"],
        "description": "Curry leaf is a tropical to sub-tropical tree native to India. It produces small, aromatic leaves that are used as a flavoring agent in many dishes.",
        "medicinal_uses": [
            "Aids digestion",
            "Helps control blood sugar levels",
            "Antioxidant properties",
            "Supports cardiovascular health",
            "Hair growth stimulant"
        ],
        "preparation": "Fresh leaves can be consumed directly or used in cooking. For medicinal purposes, leaf juice can be extracted or leaves can be dried and powdered.",
        "cautions": "Generally safe for consumption. Pregnant women should consume in moderation."
    },
    "Hibiscus Rosa-sinensis": {
        "common_names": ["Rose Mallow", "Shoe Flower"],
        "description": "Hibiscus is a genus of flowering plants in the mallow family. It features large, colorful flowers with a prominent central column of stamens.",
        "medicinal_uses": [
            "Lowers blood pressure",
            "Reduces cholesterol levels",
            "Treatment for liver disorders",
            "Anti-inflammatory properties",
            "Rich in Vitamin C"
        ],
        "preparation": "Flowers can be dried and made into a tea. Fresh petals can be consumed or used in preparations.",
        "cautions": "May interact with certain medications including chloroquine and acetaminophen."
    },
    "Ocimum Tenuiflorum (Tulsi)": {
        "common_names": ["Holy Basil", "Tulsi"],
        "description": "Ocimum tenuiflorum is an aromatic perennial plant in the family Lamiaceae. It is native to the Indian subcontinent and widespread as a cultivated plant throughout Southeast Asia.",
        "medicinal_uses": [
            "Adaptogenic stress-reliever",
            "Treatment for respiratory disorders",
            "Antibacterial and antiviral properties",
            "Supports immune system",
            "Anti-inflammatory effects"
        ],
        "preparation": "Leaves can be consumed fresh, dried, or as a tea. Essential oil can be extracted for therapeutic use.",
        "cautions": "May interact with certain medications. Not recommended in high doses during pregnancy."
    },
    #"Piper Betle": {
       # "common_names": ["Betel Leaf", "Paan"],
        #"description": "Piper betle is a perennial, dioecious creeper belonging to the Piperaceae family. It produces heart-shaped, smooth, shiny leaves with a strong aromatic flavor.",
        #"medicinal_uses": [
         #   "Oral health maintenance",
          #  "Digestive aid",
           # "Respiratory support",
            #"Wound healing properties",
            #"Antioxidant effects"
        #],
        #"preparation": "Fresh leaves can be chewed or made into a paste. Juice from leaves can be consumed for medicinal benefits.",
        #"cautions": "Should not be consumed with areca nut due to carcinogenic risks."
    #},
    "Trigonella Foenum-graecum (Fenugreek)": {
        "common_names": ["Fenugreek", "Methi"],
        "description": "Trigonella foenum-graecum is an annual plant in the family Fabaceae. It has small leaflets and white flowers that develop into pods containing small, golden-brown seeds.",
        "medicinal_uses": [
            "Helps control blood sugar levels",
            "Increases milk production in lactating mothers",
            "Reduces cholesterol levels",
            "Anti-inflammatory properties",
            "Aids digestion"
        ],
        "preparation": "Seeds can be consumed whole, sprouted, or ground into powder. Leaves can be eaten as a vegetable.",
        "cautions": "May interact with blood-thinning medications. Can cause digestive discomfort in some individuals."
    },
    "Mentha (Mint)": {
        "common_names": ["Pudina"],
        "description": "Mint is a fast-growing, aromatic herb widely used for culinary and medicinal purposes. It has a refreshing flavor and is known for its cooling properties.",
        "medicinal_uses": [
            "Aids digestion",
            "Relieves nausea",
            "Antimicrobial properties",
            "Supports respiratory health",
            "Helps in stress relief"
        ],
        "preparation": "Fresh leaves can be used in cooking, teas, or as a garnish. Mint leaves can also be dried and powdered for medicinal use.",
        "cautions": "Generally safe for consumption. Excessive use may cause acid reflux in some individuals."

    },
    "Citrus Limon (Lemon)":{
        "common_names": ["Lemon"],
        "description": "Lemon is a citrus fruit known for its tangy flavor and high vitamin C content. It is widely used in culinary, medicinal, and cosmetic applications.",
        "medicinal_uses": [
            "Boosts immunity",
            "Aids digestion",
            "Rich in antioxidants",
            "Supports heart health",
            "Helps in detoxification"
        ],
    "preparation": "Lemon juice and zest can be used in cooking, beverages, and herbal remedies. It can also be dried and powdered for medicinal use.",
    "cautions": "Generally safe for consumption. Excessive intake may cause tooth enamel erosion and acidity in some individuals."
    },
    "Amaranthus Viridis (Arive-Dantu)": {
        "common_names": ["Green Amaranth", "Slender Amaranth"],
        "description": "Amaranthus viridis is a cosmopolitan species in the botanical family Amaranthaceae. It is an annual herb with an upright, light green stem that grows to about 60–80 cm in height.",
        "medicinal_uses": [
            "Treatment of inflammation",
            "Digestive aid for constipation",
            "Helpful for respiratory disorders",
            "Treatment for anemia due to high iron content"
        ],
        "preparation": "Leaves and tender stems can be cooked as a stir-fry, added to soups, or blended into a smoothie. Fresh leaf juice can be extracted and consumed for detoxification. Crushed leaves can be applied externally for wound healing.",
        "cautions": "Excessive consumption may lead to kidney stone formation due to high oxalate content. It is advised to wash thoroughly before use to remove potential pesticide residues. Individuals with allergies to leafy greens should consume with caution."
    },
    "Artocarpus Heterophyllus (Jackfruit)": {
        "common_names": ["Jackfruit", "Kathal", "Panasa"],
        "description": "Artocarpus heterophyllus is a tropical tree species in the Moraceae family, known for producing large, spiky fruits. The tree can grow up to 20 meters tall, and its fruits are rich in fiber, vitamins, and antioxidants.",
        "medicinal_uses": [
            "Boosts immunity due to high vitamin C content",
            "Aids digestion and prevents constipation",
            "Supports heart health by reducing cholesterol levels",
            "Helps regulate blood sugar levels"
        ],
        "preparation": "Ripe fruit is eaten fresh or used in desserts, smoothies, and jams. Unripe jackfruit is cooked in curries and stir-fries as a meat substitute. Seeds can be boiled, roasted, or ground into flour for consumption.",
        "cautions": "Excess consumption may cause digestive discomfort due to high fiber content. Individuals with latex allergies should be cautious, as jackfruit contains natural latex. People with diabetes should monitor intake, as ripe jackfruit is high in natural sugars."
    },
    "Brassica Juncea (Indian Mustard)": {
        "common_names": ["Indian Mustard", "Rai", "Sarson"],
        "description": "Brassica juncea is a plant species in the Brassicaceae family, widely cultivated for its seeds, leaves, and oil. It is an annual herb with bright yellow flowers and deeply lobed leaves, growing up to 1.5 meters in height.",
        "medicinal_uses": [
            "Promotes digestion and relieves constipation",
            "Supports cardiovascular health by reducing cholesterol",
            "Has anti-inflammatory and pain-relieving properties",
            "Aids in detoxification and liver function"
        ],
        "preparation": "Leaves (mustard greens) are cooked in traditional dishes like 'Sarson da Saag' or used in salads. Seeds are used as a spice, ground into mustard paste, or extracted for oil. Mustard oil is commonly used for cooking and external application.",
        "cautions": "Excessive consumption of raw leaves may interfere with thyroid function due to goitrogens. Mustard oil should be consumed in moderation due to its high erucic acid content. People with mustard allergies should avoid seeds and oil."
    },
    "Carissa Carandas (Karanda)": {
        "common_names": ["Karanda", "Christ’s Thorn", "Karonda"],
        "description": "Carissa carandas is a hardy, evergreen shrub from the Apocynaceae family, known for its small, tart, berry-like fruits. It grows up to 3-5 meters in height and is widely cultivated in tropical and subtropical regions for its medicinal and culinary uses.",
        "medicinal_uses": [
            "Rich in vitamin C, boosts immunity",
            "Aids digestion and relieves constipation",
            "Used in traditional medicine for treating anemia",
            "Has antibacterial and antifungal properties"
        ],
        "preparation": "Fruits are consumed raw, pickled, or made into jams, chutneys, and syrups. The leaves and roots are sometimes used in herbal decoctions for medicinal purposes.",
        "cautions": "Unripe fruits contain latex, which may cause irritation in some individuals. Excessive consumption may lead to stomach discomfort due to its acidic nature. People allergic to latex should consume with caution."
    },
    
    "Plectranthus Amboinicus (Mexican Mint)": {
        "common_names": ["Mexican Mint", "Indian Borage", "Ajwain Patta", "Karpooravalli"],
        "description": "Plectranthus amboinicus is a fast-growing herb with thick, soft leaves that have a strong smell. It is used in cooking and as a natural remedy for various health problems.",
        "medicinal_uses": [
            "Helps with cough, cold, and breathing problems",
            "Improves digestion and reduces stomach pain",
            "Fights bacteria and fungal infections",
            "Boosts immunity and reduces swelling"
        ],
        "preparation": "Leaves can be chewed for sore throat relief or used in tea and herbal drinks. They are also added to soups, chutneys, and curries for flavor and health benefits.",
        "cautions": "Eating too much may cause throat irritation or allergies in some people. Pregnant and breastfeeding women should ask a doctor before using it. Too much can also cause stomach discomfort."
    },
    "Nyctanthes Arbor-tristis (Parijata)": {
    "common_names": ["Night-flowering Jasmine", "Coral Jasmine", "Harsingar"],
    "description": "Nyctanthes arbor-tristis is a small ornamental tree or shrub native to South Asia and Southeast Asia. It is known for its fragrant white flowers with orange centers that bloom at night and fall off in the morning, hence the name 'sad tree'.",
    "medicinal_uses": [
        "Treatment of fever, especially malaria",
        "Used as a remedy for arthritis and rheumatism",
        "Helps in managing constipation and digestive issues",
        "Supports liver health and detoxification"
    ],
    "preparation": "Leaves can be boiled and consumed as a decoction to reduce fever and inflammation. Flowers are sometimes used in herbal teas. Crushed leaves may be applied externally on swollen joints or inflamed areas. The bark and seeds are occasionally used in traditional formulations under guidance.",
    "cautions": "Avoid excessive intake, especially without medical supervision, due to possible hepatotoxic effects. Pregnant and lactating women should consult a physician before use. Always ensure proper identification of the plant before consumption."
   },
   "Jasminum (Jasmine)": {
    "common_names": ["Jasmine", "Chameli", "Mallige"],
    "description": "Jasminum is a genus of shrubs and vines in the olive family Oleaceae. It is widely cultivated for its beautiful and fragrant white or yellow flowers. Several species are used in traditional medicine and perfumery.",
    "medicinal_uses": [
        "Reduces stress and anxiety due to its calming aroma",
        "Used in treating skin conditions and wounds",
        "Helps in managing fever and respiratory disorders",
        "Supports oral health and alleviates mouth ulcers"
    ],
    "preparation": "Jasmine flowers are often used to make herbal tea, which helps in relaxation and respiratory relief. The essential oil extracted from the flowers can be used in aromatherapy or diluted for massage. Leaf extracts can be applied to the skin for treating wounds or rashes. Dried flowers may be used in potpourri or infused in oils.",
    "cautions": "Jasmine essential oil should be diluted before topical use to avoid skin irritation. People with fragrance sensitivity should use cautiously. Internal use of jasmine extracts should be done under medical guidance."
    },
    "Ficus Religiosa (Peepal Tree)": {
    "common_names": ["Peepal Tree", "Sacred Fig", "Ashwattha"],
    "description": "Ficus religiosa is a large, long-living deciduous tree native to the Indian subcontinent. It is considered sacred in several religions including Hinduism, Buddhism, and Jainism. The tree is characterized by heart-shaped leaves with long tapering tips and is often found near temples and public spaces.",
    "medicinal_uses": [
        "Helps manage asthma and respiratory disorders",
        "Supports treatment of diabetes by reducing blood sugar levels",
        "Used in treating diarrhea and dysentery",
        "Promotes wound healing and skin health"
    ],
    "preparation": "The bark can be boiled to prepare a decoction for managing blood sugar or digestive issues. Fresh leaves can be crushed to extract juice for external application on wounds or skin conditions. Dried leaf powder is sometimes used in traditional remedies for respiratory ailments.",
    "cautions": "Excessive consumption of bark or leaf extracts without supervision may lead to adverse effects. Pregnant and lactating women should avoid internal use unless prescribed. Always consult a healthcare provider before using for medicinal purposes."
}






    }

    


# Default information for unrecognized plants
default_plant_info = {
    "common_names": ["Unknown"],
    "description": "Information not available for this plant species.",
    "medicinal_uses": ["Consult a botanist or herbalist for accurate information"],
    "preparation": "Unknown - please consult a specialist before using any unidentified plant for medicinal purposes.",
    "cautions": "Never use unidentified plants for medicinal purposes as they may be toxic or harmful."
}

# Function definitions (as in your original code)
def hsv_mask(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_saturation_threshold = 60
    saturation_mask = cv2.inRange(hsv_image[:, :, 1], lower_saturation_threshold, 255)
    kernel_size = (5, 5)
    smoothed_mask = cv2.GaussianBlur(saturation_mask, kernel_size, 0)
    _, leaf_mask = cv2.threshold(smoothed_mask, 1, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_CLOSE, kernel)
    segmented_image = image.copy()
    segmented_image[closed_mask == 0] = [0, 0, 0]
    return segmented_image

def extract_lbp_glcm_features(image):
    lbp = feature.local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
    lbp_hist = lbp_hist.astype(float)
    lbp_hist /= (lbp_hist.sum() + 1e-6)  # Normalize

    glcm_props = []
    glcm = feature.graycomatrix((image * 255).astype(np.uint8), [1], [0], symmetric=True, normed=True)
    glcm_props.append(feature.graycoprops(glcm, prop='dissimilarity'))
    glcm_props.append(feature.graycoprops(glcm, prop='contrast'))
    glcm_props.append(feature.graycoprops(glcm, prop='homogeneity'))
    glcm_props.append(feature.graycoprops(glcm, prop='energy'))
    glcm_props.append(feature.graycoprops(glcm, prop='correlation'))
    glcm_props = np.array(glcm_props)
    glcm_props = np.squeeze(glcm_props)

    theta = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    frequency = [0.1, 0.5, 1.0]
    gabor_features = []
    for t in theta:
        for f in frequency:
            gabor_filter_real, _ = gabor(image, frequency=f, theta=t)
            gabor_features.append(np.mean(gabor_filter_real))
    gabor_features = np.array(gabor_features)
    gabor_features = np.squeeze(gabor_features)
    
    return lbp_hist, glcm_props, gabor_features

def calculate_color_moments(image):
    channels = cv2.split(image)
    color_moments = []
    for channel in channels:
        mean = np.mean(channel)
        variance = np.var(channel)
        skewness = np.mean((channel - mean) ** 3) / (variance ** (3/2) + 1e-6)
        color_moments.extend([mean, variance, skewness])
    return color_moments

def process_image(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    hsv = hsv_mask(image)
    image_rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)
    moments = calculate_color_moments(image_rgb)
    image_gray = color.rgb2gray(image)
    lbp_features, glcm_features, gabor_features = extract_lbp_glcm_features(image_gray)
    gradient_x = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    if len(gradient_magnitude.shape) != 2:
        gradient_magnitude = cv2.cvtColor(gradient_magnitude, cv2.COLOR_BGR2GRAY)
    slbp_features, sglcm_features, sgabor_features = extract_lbp_glcm_features(gradient_magnitude)
    features = np.concatenate((lbp_features, glcm_features, gabor_features, slbp_features, sglcm_features, sgabor_features, moments))
    return features, hsv, gradient_magnitude

def load_models():
    try:
        with open("Model/Best_Model.pkl", "rb") as file:
            classifier = pickle.load(file)
        with open("Model/pca.pkl", "rb") as file:
            pca = pickle.load(file)
        with open("Model/scaler.pkl", "rb") as file:
            scaler = pickle.load(file)
        return classifier, pca, scaler
    except FileNotFoundError:
        st.error("Model files not found. Please check the paths.")
        return None, None, None

# Sidebar for model selection and options
st.sidebar.header("Options")
model_path = st.sidebar.text_input("Model Directory Path", "Model")
show_processing = st.sidebar.checkbox("Show Image Processing Steps", True)

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with st.spinner("Processing image..."):
        features, segmented_image, gradient_image = process_image(image)
        
        if show_processing:
            with col2:
                st.subheader("Segmented Image")
                st.image(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB), caption="Leaf Segmentation", use_container_width=True)
                
                st.subheader("Gradient Magnitude")
                st.image(gradient_image, caption="Edge Features", use_container_width=True)
    
    classifier, pca, scaler = load_models()
    
    if classifier is not None and pca is not None and scaler is not None:
        with st.spinner("Classifying plant..."):
            X = pca.transform([features])
            X = scaler.transform(X)
            prediction = classifier.predict(X)
            plant_name = prediction[0]
            
            # Display prediction and recommendations
            st.success(f"### Identified Plant: {plant_name}")
            
            plant_details = plant_info.get(plant_name, default_plant_info)
            
            st.subheader("Medicinal Uses")
            st.write("\n".join(plant_details["medicinal_uses"]))
            
            st.subheader("Preparation")
            st.write(plant_details["preparation"])
            
            st.subheader("Cautions")
            st.write(plant_details["cautions"])
