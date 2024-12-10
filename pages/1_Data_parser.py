import streamlit as st
import requests
from bs4 import BeautifulSoup
import time
import os

from utils.parser import ensure_valid_excel_file, parse_detailed_page

# Streamlit page configuration
st.set_page_config(page_title="Krisha.kz - Almaty Parser", layout="wide")

# Dictionary of data types for parsing
PARSING_DATA_TYPES = {
    "Sale of apartments": "https://krisha.kz/prodazha/kvartiry/almaty/?page=",
    "Sale of houses": "https://krisha.kz/prodazha/doma-dachi/almaty/?page=",
    "Sale of land": "https://krisha.kz/prodazha/uchastkov/almaty/?page=",
    "Sale of commercial real estate": "https://krisha.kz/prodazha/kommercheskaya-nedvizhimost/almaty/?page=",
    "Sale of business": "https://krisha.kz/prodazha/biznes/almaty/?page=",
    "Rent of apartments": "https://krisha.kz/arenda/kvartiry/almaty/?page=",
    "Rent of houses": "https://krisha.kz/arenda/doma-dachi/almaty/?page=",
    "Commercial real estate rental": "https://krisha.kz/arenda/kommerchesky-nedvizhimost/almaty/?page=",
}

# Ensure data directory exists
data_folder = "data"
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Initialize session state
if "is_parsing" not in st.session_state:
    st.session_state.is_parsing = False
if "stop_parsing" not in st.session_state:
    st.session_state.stop_parsing = False
if "current_page" not in st.session_state:
    st.session_state.current_page = 1
if "rows_processed" not in st.session_state:
    st.session_state.rows_processed = 0

# Page layout
st.title("Krisha.kz - Almaty - Parser")

# User inputs
data_type = st.selectbox("Select the data type to parse:", list(PARSING_DATA_TYPES.keys()))
pages_to_scrape = st.number_input("Number of pages to parse:", min_value=1, max_value=500, value=10)

# Define base URL and output file
base_url = PARSING_DATA_TYPES[data_type]
excel_file = os.path.join(data_folder, f"{data_type.replace(' ', '_')}.xlsx")

# Placeholders for status and actions
status_placeholder = st.empty()
col1, col2 = st.columns(2)

# "Start parsing" button
start_button = col1.button(
    "Start Parsing",
    disabled=st.session_state.is_parsing,
    use_container_width=True
)

# "Stop parsing" button
stop_button = col2.button(
    "Stop Parsing",
    disabled=not st.session_state.is_parsing,
    use_container_width=True
)

# Handle start parsing button click
if start_button:
    st.session_state.is_parsing = True
    st.session_state.stop_parsing = False
    st.session_state.current_page = 1
    st.session_state.rows_processed = 0
    ensure_valid_excel_file(excel_file)
    status_placeholder.info("Initialization complete. Starting the parsing process...")

# Handle stop parsing button click
if stop_button:
    st.session_state.stop_parsing = True
    status_placeholder.warning("Stop request received. The process will halt after the current page is completed.")

# Parsing process
if st.session_state.is_parsing:
    page = st.session_state.current_page
    total_listings_parsed = 0

    # Progress bar for pages
    pages_progress = st.progress(0)
    # Calculate increment for each page
    increment = 1.0 / pages_to_scrape if pages_to_scrape > 0 else 1.0

    while page <= pages_to_scrape and not st.session_state.stop_parsing:
        current_url = base_url + str(page)
        response = requests.get(current_url)

        # If response is not 200, show warning and proceed to next page
        if response.status_code != 200:
            status_placeholder.warning(f"Page {page}: Received status code {response.status_code}. Skipping this page.")
            page += 1
            st.session_state.current_page = page
            pages_progress.progress(min((page / pages_to_scrape), 1.0))
            continue

        soup = BeautifulSoup(response.text, 'html.parser')
        listings = soup.find_all('div', class_='a-card')

        if not listings:
            status_placeholder.warning(f"No listings found on page {page}.")

        for listing in listings:
            data_id = listing.get('data-id')
            if data_id and not st.session_state.stop_parsing:
                success = parse_detailed_page(data_id, excel_file)
                if success:
                    st.session_state.rows_processed += 1
                    total_listings_parsed += 1
                    status_placeholder.info(
                        f"Processed listings: {st.session_state.rows_processed} | Current Page: {page}/{pages_to_scrape}"
                    )
                    # Slight delay to avoid too many requests in a short time
                    time.sleep(0.3)
            if st.session_state.stop_parsing:
                break

        # Update page and progress bar
        page += 1
        st.session_state.current_page = page
        pages_progress.progress(min((page / pages_to_scrape), 1.0))

    # Finished parsing or stopped by user
    if not st.session_state.stop_parsing:
        status_placeholder.success(
            f"Parsing completed successfully. Total processed listings: {st.session_state.rows_processed}"
        )
    else:
        status_placeholder.warning(
            f"Parsing stopped by user. Processed listings: {st.session_state.rows_processed}"
        )

    st.session_state.is_parsing = False
