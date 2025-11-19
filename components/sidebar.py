import streamlit as st

def render_sidebar():
    """
    Renders a beautiful custom sidebar for the Rail Raksha application.
    Includes app title and navigation buttons.
    """
    # Initialize session state for page navigation if not exists
    if "page" not in st.session_state:
        st.session_state["page"] = "Home"

    # Custom CSS for the sidebar
    st.markdown("""
        <style>
        /* Main theme colors */
        :root {
            --primary: #2563eb;
            --secondary: #1d4ed8;
            --accent: #60a5fa;
            --text: #f8fafc;
            --background: #0f172a;
            --hover: #3b82f6;
        }

        /* Sidebar styling */
        .css-1d391kg {
            background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
            padding: 2rem 1.5rem;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
        }

        /* Logo container */
        .logo-container {
            text-align: center;
            margin-bottom: 2.5rem;
            padding: 1.5rem;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 16px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.05);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .logo-container:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
            border-color: rgba(255, 255, 255, 0.1);
        }

        /* App title styling */
        .app-title {
            color: var(--text);
            font-size: 2.25rem;
            font-weight: 800;
            text-align: center;
            margin: 1rem 0;
            text-shadow: 0 0 20px rgba(96, 165, 250, 0.5);
            background: linear-gradient(135deg, #f8fafc, #60a5fa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.05em;
        }

        /* Navigation button styling */
        .nav-button {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            width: 100%;
            color: var(--text);
            font-size: 1.1rem;
            font-weight: 500;
            margin: 0.75rem 0;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex;
            align-items: center;
            gap: 0.75rem;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .nav-button:hover {
            background: rgba(255, 255, 255, 0.05);
            transform: translateX(5px);
            border-color: rgba(255, 255, 255, 0.1);
            box-shadow: 0 6px 16px rgba(0, 0, 0, 0.15);
        }

        .nav-button.active {
            background: linear-gradient(135deg, #2563eb, #60a5fa);
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.3);
            border: none;
        }

        /* App info styling */
        .app-info {
            margin-top: 2.5rem;
            padding-top: 1.5rem;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            color: var(--text);
            font-size: 0.9rem;
            text-align: center;
            opacity: 0.8;
        }

        .app-info p {
            margin: 0.5rem 0;
            line-height: 1.5;
        }

        /* Responsive adjustments */
        @media (max-width: 768px) {
            .app-title {
                font-size: 2rem;
            }
            
            .nav-button {
                padding: 0.875rem 1rem;
                font-size: 1rem;
            }
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        # Logo and title
        st.markdown("""
            <div class="logo-container">
                <div class="app-title">Rail Raksha</div>
            </div>
        """, unsafe_allow_html=True)

        # Navigation buttons
        pages = {
            "Home": "🚂",
            "Upload & Detect": "📁",
            "Detection Logs": "📜",
            "About": "ℹ️"
        }

        for page, icon in pages.items():
            is_active = st.session_state["page"] == page
            button_class = "nav-button active" if is_active else "nav-button"
            
            if st.button(
                f"{icon} {page}",
                key=page,
                use_container_width=True,
                help=f"Go to {page} page"
            ):
                st.session_state["page"] = page
                st.rerun()

        # App info
        st.markdown("""
            <div class="app-info">
                <p>Rail Raksha v1.0</p>
                <p>Railway Infrastructure Monitoring System</p>
            </div>
        """, unsafe_allow_html=True)

    return st.session_state["page"] 