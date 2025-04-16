document.addEventListener("DOMContentLoaded", function () {

    const button = document.getElementById("get-started-button");

    if (button) {
        button.addEventListener("click", function () {
            window.location.href = "/risk";
        });
    }

    const drugForm = document.getElementById("drug-response-form");
    if (drugForm) {
        drugForm.addEventListener("submit", function(event) {
        event.preventDefault();
        document.getElementById("display-percentage-container").style.display = "flex";
        });
    }

    const riskForm = document.getElementById("cancer-risk-form");
    if (riskForm) {
        riskForm.addEventListener("submit", function(event) {
            document.getElementById("empty-rec-text").style.display = "none";
        });
    }
    
});