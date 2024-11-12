import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Navbar with the title "Study Room" and hyperlinks
st.markdown(
    """
    <style>
        .navbar {
            background-color: #f63366;
            padding: 10px;
            text-align: center;
            color: white;
            font-size: 24px;
            font-weight: bold;
        }
        .members {
            text-align: center;
            font-size: 18px;
            margin-top: 10px;
        }
        .members a {
            color: black;
            text-decoration: none;
            margin: 0 10px;
        }
        .members a:hover {
            color: #f63366;
            text-decoration: underline;
        }
    </style>
    <div class="navbar">Study Room</div>
    <div class="members">
        <a href="https://example.com/sehyun" target="_blank">Yoo Sehyun</a> | 
        <a href="https://example.com/hyeri" target="_blank">Lee Hyeri</a> | 
        <a href="https://example.com/hyein" target="_blank">Seo Hyein</a>
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Vector Addition and Derivative Visualizer")

# Input fields for the two vectors
st.subheader("Enter the coordinates for two vectors:")
x1 = st.number_input("Vector 1 X component", value=1.0, step=0.1)
y1 = st.number_input("Vector 1 Y component", value=2.0, step=0.1)
x2 = st.number_input("Vector 2 X component", value=3.0, step=0.1)
y2 = st.number_input("Vector 2 Y component", value=-0.1, step=0.1)

# Create the vectors using NumPy
vector1 = np.array([x1, y1])
vector2 = np.array([x2, y2])
vector_sum = vector1 + vector2

# Display the resulting vector
st.write("Resultant Vector (Sum):", vector_sum)

# Visualization of Vectors
fig, ax = plt.subplots()
ax.quiver(0, 0, vector1[0], vector1[1], angles='xy', scale_units='xy', scale=1, color="r", label="Vector 1")
ax.quiver(0, 0, vector2[0], vector2[1], angles='xy', scale_units='xy', scale=1, color="b", label="Vector 2")
ax.quiver(0, 0, vector_sum[0], vector_sum[1], angles='xy', scale_units='xy', scale=1, color="g", label="Sum")

# Set plot limits and labels
max_range = max(np.abs(vector1).max(), np.abs(vector2).max(), np.abs(vector_sum).max()) + 1
ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_aspect('equal', 'box')
plt.xlabel("X")
plt.ylabel("Y")
plt.axhline(0, color='gray', linewidth=0.5)
plt.axvline(0, color='gray', linewidth=0.5)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.legend()
st.pyplot(fig)

# Function input and differentiation
st.subheader("Enter a function to differentiate:")
function_input = st.text_input("Function (in terms of x)", "x**2 + 3*x + 2")

# Parse and differentiate the function
x = sp.symbols('x')
try:
    function = sp.sympify(function_input)
    derivative = sp.diff(function, x)

    # Display the function and its derivative
    st.write("Function:", function)
    st.write("Derivative:", derivative)

    # Convert SymPy expressions to lambda functions for plotting
    func_lambda = sp.lambdify(x, function, "numpy")
    derivative_lambda = sp.lambdify(x, derivative, "numpy")

    # Define the x range for plotting
    x_vals = np.linspace(-10, 10, 400)
    y_vals = func_lambda(x_vals)
    y_deriv_vals = derivative_lambda(x_vals)

    # Plot the function and its derivative
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_vals, label="Function", color="blue")
    ax.plot(x_vals, y_deriv_vals, label="Derivative", color="orange")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Function and its Derivative")
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)

except (sp.SympifyError, TypeError):
    st.error("Invalid function input. Please enter a valid mathematical expression.")
