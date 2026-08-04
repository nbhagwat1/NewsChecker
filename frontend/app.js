// Used for verifying that the backend API is reachable during development.
async function testAPI() {
    const response = await fetch("http://127.0.0.1:8000/");
    const data = await response.json();
    console.log(data)
}

/*
 * Sends article text to the backend for classification,
 * then updates the UI with the model's confidence score
 * and trustworthiness label.
 */
async function checkArticle() {
    const articleText = document.querySelector(".article-text-input").value;

    // Prevent unnecessary API requests when no article text is provided.
    if (articleText === "") {
        document.querySelector(".result-label").textContent = "The title and the actual text of the article are required.";
        document.querySelector(".result-label").style.color = "rgb(255, 0, 0)";
        document.querySelector(".result-score").textContent = "";
        return;
    }

    document.querySelector(".result-label").textContent = "Processing...";
    document.querySelector(".result-label").style.color = "rgb(212, 255, 0)";
    document.querySelector(".result-score").textContent = "";

    // Send article text to the FastAPI backend for ML inference.
    const result = await fetch("http://127.0.0.1:8000/check", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            text: articleText
        })
    });

    const resultData = await result.json();

    document.querySelector(".result-score").textContent = "Score: " + resultData.score;

    let label = "UNCERTAIN";
    const resultScore = resultData.score;

    const labelElement = document.querySelector(".result-label");
    labelElement.style.color = "rgb(212, 255, 0)";

    // Convert the model confidence score into a user-friendly category.
    // Thresholds correspond to confidence ranges defined by the application.
    if (resultScore >= 0.8 && resultScore <= 1.0) {
        label = "LIKELY TRUSTWORTHY";
        labelElement.style.color = "rgb(0, 143, 21)";
    } else if (resultScore >= 0.6 && resultScore < 0.8) {
        label = "PROBABLY TRUSTWORTHY";
        labelElement.style.color = "rgb(0, 255, 38)";
    } else if (resultScore >= 0.2 && resultScore < 0.4) {
        label = "PROBABLY UNTRUSTWORTHY";
        labelElement.style.color = "rgb(255, 111, 0)";
    } else if (resultScore >= 0.0 && resultScore < 0.2) {
        label = "LIKELY UNTRUSTWORTHY";
        labelElement.style.color = "rgb(255, 0, 0)";
    }

    labelElement.textContent = label

    const displayElement = document.querySelector(".result-display")
    displayElement.style.padding = "8px";
}