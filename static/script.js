document.addEventListener("DOMContentLoaded", function() {
    const form = document.getElementById("prediction-form");
    const resultDiv = document.getElementById("result");
    const randomizeButton = document.getElementById("randomize-btn");

    // // Randomize button functionality
    // randomizeButton.addEventListener("click", function() {
    //     // Get all input elements from the form
    //     const inputs = form.querySelectorAll("input[type='number']");

    //     inputs.forEach(input => {
    //         // Generate random values for each input
    //         if (input.id === 'latitude' || input.id === 'longitude') {
    //             // Randomize latitude and longitude within valid ranges
    //             input.value = (Math.random() * (90 - (-90)) + (-90)).toFixed(6); // Latitude [-90, 90]
    //             if (input.id === 'longitude') {
    //                 input.value = (Math.random() * (180 - (-180)) + (-180)).toFixed(6); // Longitude [-180, 180]
    //             }
    //         } else if (input.id === 'stars') {
    //             // Randomize star rating within the range of 1 to 5
    //             input.value = (Math.random() * (5 - 1) + 1).toFixed(1); // Stars [1, 5]
    //         } else {
    //             // Randomize other numeric inputs in the range of 0 to 100
    //             input.value = Math.floor(Math.random() * 101);
    //         }
    //     });
    // });

    // Form submission handling
    form.addEventListener("submit", function(event) {
        event.preventDefault();  // Prevent the default form submission

        // Gather all the form data
        const formData = new FormData(form);
        
        // Convert the FormData to a plain object
        const data = {};
        formData.forEach((value, key) => {
            data[key] = value;
        });

        // Display a loading message while waiting for the result
        resultDiv.innerHTML = "Making prediction... Please wait.";

        // Send data to the server via fetch
        fetch("/predict_adaboost", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify(data), // Send the data as JSON
        })
        .then(response => response.json())
        .then(result => {
            // Display the prediction result (assumed result format: { "prediction": value })
            if (result && result.prediction) {
                resultDiv.innerHTML = `Prediction Result: ${result.prediction}`;
            } else {
                resultDiv.innerHTML = "Error: Unable to retrieve prediction.";
            }
        })
        .catch(error => {
            console.error("Error during prediction:", error);
            resultDiv.innerHTML = "Error: Something went wrong. Please try again.";
        });
    });
});
