# Object Detection and Classification

This project is an advanced AI vision analytics application that performs real-time object detection and image classification using TensorFlow.js and pre-trained models. The application displays the camera feed, detects objects, classifies images, and provides various visualizations and statistics.

## Features

- Real-time object detection using the COCO-SSD model
- Real-time image classification using the MobileNet model
- Displays FPS and processing time per frame
- Visualizes detection metrics, object distribution, and confidence distribution
- Provides detailed statistics and visualizations for detected objects

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/AshishSingh0311/Object-dectition-.git
    cd Object-dectition-
    ```

2. Install the dependencies:
    ```sh
    npm install
    ```

3. Start the development server:
    ```sh
    npm run dev
    ```

4. Open the application in your browser:
    - The terminal should display a URL (usually `http://localhost:3000`). Open this URL in your web browser.

## Usage

- The application will access your camera and display the feed.
- It will detect objects and classify images in real-time.
- Various statistics and visualizations will be updated in real-time.

## Project Structure

- `src/App.tsx`: The main application component that handles object detection, image classification, and rendering the UI.
- `public/index.html`: The HTML template for the application.
- `src/index.tsx`: The entry point for the React application.

## Dependencies

- `@tensorflow/tfjs`: TensorFlow.js library for running machine learning models in the browser.
- `@tensorflow-models/coco-ssd`: Pre-trained COCO-SSD model for object detection.
- `@tensorflow-models/mobilenet`: Pre-trained MobileNet model for image classification.
- `react`: JavaScript library for building user interfaces.
- `recharts`: Library for building charts and visualizations.
- `lucide-react`: Icon library for React.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- TensorFlow.js team for providing the pre-trained models and libraries.
- Recharts team for the charting library.
- Lucide team for the icon library.

## Author

- Ashish Vivek Singh
