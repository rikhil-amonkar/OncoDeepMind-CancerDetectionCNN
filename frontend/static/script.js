document.addEventListener("DOMContentLoaded", function () {
    const button = document.getElementById("get-started-button");

    if (button) {
        button.addEventListener("click", function () {
            window.location.href = "/risk";
        });
    }
});

document.getElementById("drug-response-form").addEventListener("submit", function(event) {
    event.preventDefault();

    document.getElementById("display-percentage-container").style.display = "block";

});

document.getElementById("cancer-risk-form").addEventListener("submit", function(event) {
    event.preventDefault();

    document.getElementById("empty-rec-text").style.display = "none";

});