import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Dinosaur Image Classifier",
    page_icon="🦕",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Function to preprocess image
def preprocess_image(image, target_size=(150, 150)):
    # Convert to RGB if image is in RGBA
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    
    # Resize image
    img = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# Function to load model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('model/dinosaur_classifier_final.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Class names (update these according to your model's classes)
DINOSAUR_CLASSES = [
    "Allosaurus",
    "Ankylosaurus",
    "Brachiosaurus",
    "Carnotaurus",
    "Parasaurolophus",
    "Pteranodon",
    "Spinosaurus",
    "Stegosaurus",
    "Triceratops",
    "Tyrannosaurus Rex",
    "Velociraptor"
]

# Function to make prediction
def predict_dinosaur(model, image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Get prediction
    predictions = model.predict(processed_image)
    
    # Get the highest probability and its index
    max_prob = np.max(predictions[0])
    predicted_class = DINOSAUR_CLASSES[np.argmax(predictions[0])]
    
    # Get top 3 predictions
    top_3_idx = predictions[0].argsort()[-3:][::-1]
    top_3_predictions = [
        (DINOSAUR_CLASSES[idx], float(predictions[0][idx]) * 100)
        for idx in top_3_idx
    ]
    
    return max_prob, predicted_class, top_3_predictions

def main():
    st.title("🦕 Dinosaur Image Classifier")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload an Image")
        uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            try:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Load model
                model = load_model()
                
                if model is None:
                    st.error("Failed to load the model. Please try again.")
                    return
                
                # Add prediction button
                if st.button("Analyze Image"):
                    with st.spinner("Analyzing..."):
                        # Make prediction
                        confidence, predicted_class, top_3_predictions = predict_dinosaur(model, image)
                        
                        # Display results in the second column
                        with col2:
                            st.markdown("### Analysis Results")
                            
                            # Check if it's likely a dinosaur (you can adjust the threshold)
                            if confidence > 0.5:  # 50% confidence threshold
                                st.success("✅ This image appears to be a dinosaur!")
                                st.markdown(f"### Predicted Species: {predicted_class}")
                                
                                # Display top 3 predictions with confidence bars
                                st.markdown("### Top 3 Matches:")
                                for species, conf in top_3_predictions:
                                    st.markdown(f"**{species}**")
                                    st.progress(conf/100)
                                    st.markdown(f"Confidence: {conf:.2f}%")
                                    st.markdown("---")
                            else:
                                st.error("❌ This image doesn't appear to be a dinosaur.")
                                st.markdown("Please upload a clear image of a dinosaur.")
                
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
                st.markdown("Please try uploading a different image.")
    
    # Information section
    st.markdown("---")
    with st.expander("ℹ️ About this Classifier"):
        st.markdown("""
        This image classifier can:
        1. Determine if an uploaded image contains a dinosaur
        2. Identify the specific species of dinosaur
        3. Provide confidence scores for the top matches
        
        Supported dinosaur species:
        """)
        # Display supported species in columns
        species_cols = st.columns(3)
        for idx, species in enumerate(DINOSAUR_CLASSES):
            species_cols[idx % 3].markdown(f"- {species}")
        
        st.markdown("""
        For best results:
        - Upload clear, well-lit images
        - Ensure the dinosaur is the main subject
        - Use images with minimal background clutter
        """)

if __name__ == "__main__":
    main()