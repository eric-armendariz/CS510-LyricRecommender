# Lyric-Based Song Recommender

Welcome to LyRecs, our recommendations-by-lyrics application!

## Getting Started

To get the application up and running on your local machine, follow these steps:

### Prerequisites

Ensure you have Node.js and Python installed on your computer. 

### Starting the Frontend

1. Navigate to the `lyrecs` directory:
    ```bash
    cd lyrecs
    ```
2. Run the following command to start the frontend:
    ```bash
    npm start
    ```
   This command will launch the application on `localhost:3000`.

### Starting the Backend

1. Ensure you are in the root directory of the project.
2. Start the backend by running:
    ```bash
    python app.py
    ```

### Using the Application

Once both the frontend and backend are up and running:

- Paste the lyrics of the song you're interested in into the provided text box.
- Click the `Submit` button.
- The LSA model, which has been preprocessed and trained, will handle your query and display 10 song recommendations along with their titles and artists.

Enjoy discovering new music with LyRecs!
