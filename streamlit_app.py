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

# Define the page navigation
page = st.radio("Go to page:", ["1", "2", "3"], horizontal=True)

# Page 1: Vector Addition and Differentiation
if page == "1":
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

# Page 2: Function Integration
elif page == "2":
    st.title("Function Integration Visualizer")

    # Input for the function to integrate
    st.subheader("Enter a function to integrate:")
    function_input = st.text_input("Function (in terms of x)", "x**2 + 3*x + 2")

    # Limits for definite integration
    st.subheader("Define the limits for integration:")
    lower_limit = st.number_input("Lower limit", value=0.0, step=0.1)
    upper_limit = st.number_input("Upper limit", value=5.0, step=0.1)

    # Parse and integrate the function
    x = sp.symbols('x')
    try:
        function = sp.sympify(function_input)
        integral = sp.integrate(function, x)
        definite_integral = sp.integrate(function, (x, lower_limit, upper_limit))

        # Display the function, its indefinite integral, and the definite integral
        st.write("Function:", function)
        st.write("Indefinite Integral:", integral)
        st.write(f"Definite Integral from {lower_limit} to {upper_limit}:", definite_integral)

        # Convert SymPy expression to lambda function for plotting
        func_lambda = sp.lambdify(x, function, "numpy")
        integral_lambda = sp.lambdify(x, integral, "numpy")

        # Define the x range for plotting
        x_vals = np.linspace(lower_limit, upper_limit, 400)
        y_vals = func_lambda(x_vals)
        y_integral_vals = integral_lambda(x_vals)

        # Plot the function and its integral
        fig, ax = plt.subplots()
        ax.plot(x_vals, y_vals, label="Function", color="blue")
        ax.plot(x_vals, y_integral_vals, label="Indefinite Integral", color="green")
        plt.fill_between(x_vals, y_vals, color="blue", alpha=0.1, label="Area under curve")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Function and its Integral")
        plt.legend()
        plt.grid(True)
        st.pyplot(fig)

    except (sp.SympifyError, TypeError):
        st.error("Invalid function input. Please enter a valid mathematical expression.")

# Page 3: Eigenvalues and Eigenvectors
elif page == "3":
    st.title("Eigenvalues and Eigenvectors of a 2x2 Matrix")

    # Input fields for a 2x2 matrix
    st.subheader("Enter the values for a 2x2 matrix:")
    a = st.number_input("Matrix element a (top-left)", value=1.0, step=0.1)
    b = st.number_input("Matrix element b (top-right)", value=2.0, step=0.1)
    c = st.number_input("Matrix element c (bottom-left)", value=3.0, step=0.1)
    d = st.number_input("Matrix element d (bottom-right)", value=4.0, step=0.1)

    # Create the matrix
    matrix = np.array([[a, b], [c, d]])

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    # Display the matrix, eigenvalues, and eigenvectors
    st.write("Matrix:")
    st.write(matrix)
    st.write("Eigenvalues:", eigenvalues)
    st.write("Eigenvectors:", eigenvectors)

    # Plot eigenvectors with the eigenvalues as scaling factors
    fig, ax = plt.subplots()
    origin = [0, 0]  # origin point for the vectors

    # Plot eigenvectors scaled by eigenvalues
    for i in range(len(eigenvalues)):
        eigenvector = eigenvectors[:, i]
        eigenvalue = eigenvalues[i]
        ax.quiver(*origin, eigenvector[0], eigenvector[1], angles='xy', scale_units='xy', scale=1, color="r" if i == 0 else "b", label=f"Eigenvector {i+1}")

    # Set plot limits
    limit = max(abs(eigenvalues).max(), abs(matrix).max()) * 1.5
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', 'box')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Eigenvalues and Eigenvectors")
    plt.axhline(0, color='gray', linewidth=0.5)
    plt.axvline(0, color='gray', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.legend()
    st.pyplot(fig)
