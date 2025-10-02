# ================================================================
# ?? AI-Powered Warehouse Optimization App (Streamlit)
# ================================================================
# ?? Route Optimization (Travelling Salesman Problem)
# ?? Shelf Space Allocation (K-Means Clustering)
# ?? Inventory Management with Transactions
# ?? CSV Upload Support
# ================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.cluster import KMeans

# Set Streamlit page configuration
st.set_page_config(page_title="Warehouse AI Optimizer", layout="centered")

# Sidebar for navigation
st.sidebar.title("?? Warehouse AI App")
tab = st.sidebar.radio("Go to", ["?? Home", "?? Route Optimizer", "?? Shelf Clustering", "?? Inventory Tracker"])

# ---------------------------------------------
# ?? HOME
# ---------------------------------------------
if tab == "?? Home":
    st.title("?? AI-Powered Warehouse Optimization")
    st.markdown("""
    This interactive app demonstrates how AI can optimize:
    - **Delivery Routes** (TSP)
    - **Shelf Space Allocation** (K-Means Clustering)
    - **Inventory Tracking** (Real-time updates)

    Built with **Python**, **Pandas**, **Scikit-learn**, **Matplotlib**, and **Streamlit**.
    """)

# ---------------------------------------------
# ?? ROUTE OPTIMIZER (Travelling Salesman Problem)
# ---------------------------------------------
elif tab == "?? Route Optimizer":
    st.title("?? Route Optimizer – TSP")

    # Coordinates for warehouse and customers
    locations = {
        "Warehouse": (0, 0),
        "Customer A": (2, 4),
        "Customer B": (5, 2),
        "Customer C": (6, 6),
        "Customer D": (8, 3)
    }

    # Extract coordinates and labels
    points = np.array(list(locations.values()))
    labels = list(locations.keys())

    # TSP algorithm (Greedy Nearest Neighbor)
    def tsp_greedy(points, start=0):
        n = len(points)
        visited = [False] * n
        route = [start]
        visited[start] = True
        total_dist = 0
        for _ in range(n - 1):
            last = route[-1]
            nearest = None
            nearest_dist = float('inf')
            for i in range(n):
                if not visited[i]:
                    dist = np.linalg.norm(points[last] - points[i])
                    if dist < nearest_dist:
                        nearest = i
                        nearest_dist = dist
            route.append(nearest)
            visited[nearest] = True
            total_dist += nearest_dist
        total_dist += np.linalg.norm(points[route[-1]] - points[start])
        route.append(start)
        return route, total_dist

    route, total_distance = tsp_greedy(points)

    # Show Route
    st.subheader("?? Optimized Route")
    for idx in route:
        st.write("??", labels[idx])

    st.write(f"?? **Total Distance:** `{round(total_distance, 2)} units`")

    # Visualize Route
    st.subheader("?? Route Map")
    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(len(route) - 1):
        p1, p2 = points[route[i]], points[route[i+1]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b--')
    ax.scatter(points[:, 0], points[:, 1], c='red', s=100)
    for i, label in enumerate(labels):
        ax.text(points[i][0]+0.1, points[i][1]+0.1, label, fontsize=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Optimized Delivery Route")
    st.pyplot(fig)

# ---------------------------------------------
# ?? SHELF CLUSTERING (K-Means)
# ---------------------------------------------
elif tab == "?? Shelf Clustering":
    st.title("?? Shelf Space Allocation – K-Means")

    # Sample product data
    data = {
        'Product': ['A','B','C','D','E','F','G','H','I','J'],
        'Demand_Frequency': [200, 50, 300, 120, 80, 400, 20, 180, 260, 95],
        'Size': [3, 2, 5, 4, 3, 6, 2, 4, 5, 3],
    }
    df = pd.DataFrame(data)

    st.subheader("?? Product Table")
    st.dataframe(df)

    # Clustering
    X = df[['Demand_Frequency', 'Size']]
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
    df['Shelf_Zone'] = kmeans.fit_predict(X)

    st.subheader("?? Shelf Zones")
    st.dataframe(df)

    # Plot
    fig2 = px.scatter(df, x='Demand_Frequency', y='Size',
                      color='Shelf_Zone', text='Product',
                      title="Shelf Allocation (K-Means)",
                      color_continuous_scale='Portland')
    st.plotly_chart(fig2)

# ---------------------------------------------
# ?? INVENTORY TRACKER (with CSV Upload)
# ---------------------------------------------
elif tab == "?? Inventory Tracker":
    st.title("?? Inventory Tracker")

    st.markdown("#### ?? Upload Inventory CSV (Optional)")
    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    if uploaded_file is not None:
        df_inv = pd.read_csv(uploaded_file)
    else:
        df_inv = pd.DataFrame({
            "Product_ID": ["P101", "P102", "P103", "P104", "P105"],
            "Product_Name": ["Sugar", "Rice", "Oil", "Tea", "Biscuits"],
            "Location": ["Shelf A", "Shelf B", "Shelf C", "Shelf D", "Shelf A"],
            "Status": ["IN", "IN", "OUT", "IN", "IN"],
            "Quantity": [50, 30, 0, 20, 100]
        })

    st.markdown("#### ?? Current Inventory")
    st.dataframe(df_inv)

    st.markdown("#### ?? Update Stock")

    with st.form("stock_form"):
        product_id = st.selectbox("Select Product", df_inv["Product_ID"])
        change = st.number_input("Quantity Change (+/-)", value=0, step=1)
        status = st.selectbox("Status", ["IN", "OUT"])
        update = st.form_submit_button("Apply Update")

        if update:
            idx = df_inv[df_inv["Product_ID"] == product_id].index[0]
            df_inv.at[idx, "Quantity"] += change
            df_inv.at[idx, "Status"] = status
            st.success(f"? Updated stock for {product_id}")

    st.markdown("#### ?? Updated Inventory")
    st.dataframe(df_inv)

    st.markdown("#### ?? Inventory Bar Chart")
    fig3 = px.bar(df_inv, x="Product_Name", y="Quantity", color="Status",
                  title="Stock Level per Product", text="Quantity")
    st.plotly_chart(fig3)
