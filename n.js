require("dotenv").config();
const express = require("express");
const { HfInference } = require("@huggingface/inference");
const cors = require("cors");

// Initialize Hugging Face Inference with the API key from environment variables
const hf = new HfInference(process.env.HUGGING_FACE_API_KEY);

// Create an instance of Express
const app = express();

// Middleware to parse JSON
app.use(express.json());

// Enable CORS for all origins (you can restrict it to specific origins in production)
app.use(cors());

// Root Endpoint
app.get("/", (req, res) => {
  res.send("Reached root");
});

// Function to create the prompt for the question-answering task
const createPrompt = (question, context) => {
  return `Write a program using loops or recursion to solve: question: ${question} context: ${context}`;
};

// Endpoint to handle medical queries
app.post("/askmedical", async (req, res) => {
  console.log("Received medical query request");

  try {
    const { question, context } = req.body;

    // Validate the inputs
    if (!question || !context) {
      return res.status(400).json({
        error: "Both 'question' and 'context' fields are required.",
      });
    }

    // Log the received inputs for debugging
    console.log(`Question: ${question}`);
    console.log(`Context: ${context}`);

    // Create the prompt using the updated function
    const prompt = createPrompt(question, context);

    // Call Hugging Face API with your model
    const result = await hf.textGeneration({
      model: "Bandipreethamreddy/train_model",  // Replace with your fine-tuned model name
      inputs: prompt,
      parameters: {
        max_new_tokens: 230,  // Allow more tokens for a detailed response
        temperature: 0.5,     // Lower temperature for deterministic results
        top_p: 0.95,
        do_sample: false,     // Set to false to avoid randomness and get focused answers
      },
    });

    // Check if the result is valid
    if (!result || !result.generated_text) {
      return res.status(500).json({ error: "Failed to generate a valid response." });
    }

    // Log and send the result
    console.log("Generated Result:", result);
    res.json({ answer: result.generated_text.trim() });
  } catch (error) {
    console.error("Error processing request:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

// Start the Server
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
  console.log(`Medical Query API server is listening on port ${PORT}`);
});





