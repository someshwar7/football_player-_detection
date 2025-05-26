import streamlit as st
import plotly.express as px
import pandas as pd

# Custom CSS
st.markdown("""
<style>
/* Global styles */
body {
    font-family: 'Arial', sans-serif;
    background-color: #f4f4f9;
}

h1, h2 {
    color: #2c3e50;
    text-align: center;
}

.container {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
    padding: 20px;
}

.box {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    text-align: center;
}

.custom-button {
    background-color: #4CAF50;
    color: white;
    padding: 10px 20px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s ease;
}

.custom-button:hover {
    background-color: #45a049;
}

/* Modal styles */
.modal {
    display: none;
    position: fixed;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background-color: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}

.modal.show {
    display: block;
}

/* Button container */
button-container {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Title
st.title('Interactive Streamlit + HTML/CSS App')

# Create a form with interactive widgets
name = st.text_input('Enter your name')
age = st.number_input('Enter your age', min_value=1, max_value=100)
gender = st.selectbox('Select your gender', ['Male', 'Female', 'Other'])

# Custom submit button
if st.button('Submit', key='submit_button'):
    st.write(f"Hello {name}, you are {age} years old and identify as {gender}.")

# Custom Hover Effects Button
st.markdown("""
<div style="text-align:center;">
    <button class="custom-button" onclick="alert('Thanks for clicking!')">Click Me!</button>
</div>
""", unsafe_allow_html=True)

# Creating a simple dataframe for an interactive chart
data = pd.DataFrame({
    "Category": ["A", "B", "C", "D", "E"],
    "Value": [10, 20, 30, 40, 50]
})

# Interactive Chart
fig = px.bar(data, x="Category", y="Value", title="Category vs Value", color="Category")
st.plotly_chart(fig)

# Custom Modal Popup
st.markdown("""
<div id="modal" class="modal">
    <h2>Welcome to the App!</h2>
    <p>This is an interactive modal popup.</p>
    <button onclick="document.getElementById('modal').classList.remove('show')">Close</button>
</div>
<script>
setTimeout(function() {
    document.getElementById('modal').classList.add('show');
}, 1000);
</script>
""", unsafe_allow_html=True)

# Grid Layout for content
st.markdown("""
<div class="container">
    <div class="box">
        <h2>Box 1</h2>
        <p>This is box 1 content.</p>
    </div>
    <div class="box">
        <h2>Box 2</h2>
        <p>This is box 2 content.</p>
    </div>
    <div class="box">
        <h2>Box 3</h2>
        <p>This is box 3 content.</p>
    </div>
</div>
""", unsafe_allow_html=True)
