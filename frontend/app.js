async function testAPI() {
    const response = await fetch("http://127.0.0.1:8000/");
    const data = await response.json();
    console.log(data)
}

async function checkArticle() {
    const articleText = document.querySelector(".article-text-input").value;

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

    document.querySelector(".result-score").textContent = resultData.score;
}