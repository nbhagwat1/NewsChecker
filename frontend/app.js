async function testAPI() {
    const response = await fetch("http://127.0.0.1:8000/");
    const data = await response.json();
    console.log(data)
}

async function checkArticle() {
    const articleText = document.querySelector(".article-text-input").value;

    if (articleText === "") {
        document.querySelector(".result-label").textContent = "The title and the actual text of the article are required.";
        document.querySelector(".result-label").style.color = "rgb(255, 0, 0)";
        document.querySelector(".result-score").textContent = "";
        return;
    }

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