import gradio as gr
import tensorflow as tf
import numpy as np

from keras.src.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.platypus.flowables import HRFlowable
from datetime import datetime


# Load Model

model = tf.keras.models.load_model("brain_tumor_mri_final.keras")

# Classes
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

IMG_SIZE = 224


# create MRI REPORT

def create_report_mri(predicted_class, confidence):

    pdf_path = "mri_report.pdf"

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=30
    )

    styles = getSampleStyleSheet()
    elements = []

    # ===== Title =====
    title = Paragraph(
        "<font size=22 color='darkblue'><b>Brain MRI Analysis Report</b></font>",
        styles['Title']
    )

    elements.append(title)
    elements.append(Spacer(1, 12))

    # ===== Line =====
    elements.append(HRFlowable(width="100%", color=colors.darkblue))
    elements.append(Spacer(1, 20))

    # ===== Date =====
    date_text = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    date_para = Paragraph(
        f"<b>Generated On:</b> {date_text}",
        styles['Normal']
    )

    elements.append(date_para)
    elements.append(Spacer(1, 20))

    # ===== Prediction Result =====
    result_title = Paragraph(
        "<font size=16 color='green'><b>Prediction Result</b></font>",
        styles['Heading2']
    )

    elements.append(result_title)
    elements.append(Spacer(1, 10))

    # ===== Table =====
    data = [
        ["Parameter", "Value"],
        ["Predicted Tumor Type", predicted_class.upper()],
        ["Confidence Score", f"{confidence * 100:.2f}%"]
    ]

    table = Table(data, colWidths=[220, 220])

    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),

        ('BACKGROUND', (0, 1), (-1, -1), colors.whitesmoke),

        ('GRID', (0, 0), (-1, -1), 1, colors.black),

        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),

        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),

        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),

        ('BACKGROUND', (0, 1), (-1, 1), colors.lightblue),

        ('BACKGROUND', (0, 2), (-1, 2), colors.beige),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 25))

    # ===== Recommendation =====
    recommendation = Paragraph(
        """
        <font size=12>
        <b>Medical Note:</b><br/>
        This AI-generated report is intended for research and educational purposes only.
        Please consult a certified radiologist or neurologist for professional diagnosis.
        </font>
        """,
        styles['BodyText']
    )

    elements.append(recommendation)
    elements.append(Spacer(1, 20))

    # ===== Footer =====
    footer = Paragraph(
        "<font color='grey'>Generated using Deep Learning VGG16 MRI Classification System</font>",
        styles['Italic']
    )

    elements.append(footer)

    # Build PDF
    doc.build(elements)

    return pdf_path

def predict_image(img_path):
  """Predict the tumor class for a single image file, robust to RGBA/grayscale input."""
  # # Convert to RGB
  image = img_path.convert("RGB")
  # Resize
  image = image.resize((IMG_SIZE, IMG_SIZE))
  img_array = img_to_array(image)
  img_array = np.expand_dims(img_array, axis=0)

  # Use the SAME preprocessing as training -- this is critical for correct predictions
  img_array = preprocess_input(img_array)
  prediction = model.predict(img_array,verbose=0)[0]
  predicted_class = class_names[np.argmax(prediction)]
  confidence = float(np.max(prediction))
  
  # MRI REPORT
  report=create_report_mri(predicted_class,confidence)
  return ({
        predicted_class:confidence
    # class_names[i]: float(prediction[i])
    # for i in range(len(class_names))
    }, report)


interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Label(num_top_classes=4),
        gr.File(label="Download Report")
    ],
    title="Brain Tumor Classification",
    description="Classify MRI images into Glioma, Meningioma, Pituitary, or No Tumor using VGG16 Transfer Learning.",

)

interface.launch()