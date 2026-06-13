const API_URL =
"https://customer-churn-intelligence-platform.onrender.com";

document.addEventListener("DOMContentLoaded", () => {

    loadDashboardStats();
    loadCharts();

    document
        .getElementById("predictionForm")
        .addEventListener(
            "submit",
            predictCustomer
        );

});

async function loadDashboardStats() {

    try {

        const response = await fetch(
            `${API_URL}/dashboard-stats`
        );

        const data = await response.json();
        



        document.getElementById(
            "totalCustomers"
        ).innerText =
            data.total_customers;

        document.getElementById(
            "revenueRisk"
        ).innerText =
            "$" + data.revenue_at_risk.toLocaleString();

        document.getElementById(
            "cltv"
        ).innerText =
            "$" + data.avg_cltv.toLocaleString();

    }

    catch (error) {

        console.error(
            "Dashboard Stats Error:",
            error
        );

    }

}

async function predictCustomer(event) {

    event.preventDefault();

    const payload = {

        Age: parseInt(
            document.getElementById("age").value
        ),

        TenureinMonths: parseInt(
            document.getElementById("tenure").value
        ),

        MonthlyCharge: parseFloat(
            document.getElementById("monthlyCharge").value
        ),

        Contract:
            document.getElementById("contract").value,

        InternetType:
            document.getElementById("internetType").value,

        PaymentMethod:
            document.getElementById("paymentMethod").value,

        SatisfactionScore: parseInt(
            document.getElementById("satisfaction").value
        ),

        Number_of_Referrals: parseInt(
            document.getElementById("referrals").value
        )
    };

    try {

        const response = await fetch(
            `${API_URL}/predict`,
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify(payload)
            }
        );

        const result = await response.json();

        document.getElementById(
            "riskScore"
        ).innerText =
            result.churn_probability + "%";

        document.getElementById(
            "churnRate"
        ).innerText =
            result.churn_probability + "%";

        document.getElementById(
            "riskDrivers"
        ).innerHTML =
            result.top_risk_drivers
                .map(item => `<li>${item}</li>`)
                .join("");

        const riskElement =
            document.getElementById("riskLevel");

        riskElement.innerText =
            result.risk_level;

        riskElement.className = "badge";

        if (
            result.risk_level.toLowerCase() === "high"
        ) {
            riskElement.classList.add("high");
        }

        else if (
            result.risk_level.toLowerCase() === "medium"
        ) {
            riskElement.classList.add("medium");
        }

        else {
            riskElement.classList.add("low");
        }

        document.getElementById(
            "revenueRisk"
        ).innerText =
            "$" +
            (
                payload.MonthlyCharge *
                payload.TenureinMonths
            ).toFixed(0);

        document.getElementById(
            "cltv"
        ).innerText =
            "$" +
            (
                payload.MonthlyCharge *
                payload.TenureinMonths * 1.5
            ).toFixed(0);

    }

    catch (error) {

        console.error(error);

        alert("Prediction failed.");

    }

}

function loadCharts() {

    const featureCtx =
        document
            .getElementById(
                "featureChart"
            )
            .getContext("2d");

    new Chart(featureCtx, {

        type: "bar",

        data: {

            labels: [
                "Tenure",
                "Contract",
                "Monthly Charge",
                "Internet Type",
                "CLTV"
            ],

            datasets: [{

                label:
                    "Feature Importance",

                data: [
                    32,
                    26,
                    18,
                    15,
                    9
                ]

            }]
        }

    });

    const riskCtx =
        document
            .getElementById(
                "riskChart"
            )
            .getContext("2d");

    new Chart(riskCtx, {

        type: "doughnut",

        data: {

            labels: [
                "Low Risk",
                "Medium Risk",
                "High Risk"
            ],

            datasets: [{

                data: [
                    45,
                    30,
                    25
                ]

            }]
        }

    });

}
