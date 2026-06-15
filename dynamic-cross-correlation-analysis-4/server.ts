import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = process.env.PORT ? parseInt(process.env.PORT, 10) : 3000;

app.use(express.json({ limit: "50mb" }));

// Initialize Gemini Client safely
let ai: GoogleGenAI | null = null;
function getGemini(): GoogleGenAI {
  if (!ai) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable is missing. Configure it in Settings > Secrets.");
    }
    ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return ai;
}

// REST APIs
app.get("/api/health", (req, res) => {
  res.json({ status: "ok", time: new Date().toISOString() });
});

// Endpoint: AI Correlation Insights calling Gemini
app.post("/api/gemini/insights", async (req, res) => {
  try {
    const { correlationData, params, model } = req.body;
    const client = getGemini();
    const modelToUse = model || "gemini-3.5-flash";

    const systemPrompt = `You are a world-class computational biophysics and molecular dynamics interpreter. 
You will examine residue cross-correlation statistics (mean correlation and std, positive/negative probabilities) from dynamic cross-correlation (DCC) matrices of biomolecular simulations.
Provide a professional, rigorous, and structural analysis.
Format your response as a structured JSON object with the exact fields described below. DO NOT include any markdown blocks of JSON, just make sure it parses as valid JSON on our backend.
Fields:
- "summary": A detailed high-level summary of the dynamical coupling, hinges, or allosteric networks detected in the dataset. (Markdown string)
- "predictions": Array of 3 to 5 predicted highly correlated/coupled residue pairs. For each, include:
    - "pair": "Residue i - Residue j"
    - "explanation": "Biophysical and structural explanation of their potential interaction, binding interface, or secondary structure stabilization."
    - "confidence": A percentage score (e.g., "92%") representing confidence in the coupling.
- "anomalies": Array of 2 to 4 anomalous residue interactions (e.g. unexpected high/low correlation given distance, high standard deviation indicating bistable/conformation switching). For each, include:
    - "pair": "Residue i - Residue j"
    - "detection": "Analytical detail on why this is anomalous (e.g. high variation, long-range direct correlation without intermediating sequence)."
    - "implication": "Conformational, hinge, allosteric regulation, or interface binding implications."
    - "confidence": "85%"
- "pairRankings": Array of 5 most critical residue-pairs ranked by structural importance. For each, include:
    - "rank": 1 (number)
    - "pair": "Residue i - Residue j"
    - "role": "Hinge point, binding interface, alpha-helix backbone, flexible loop stabilizer, etc."
    - "score": 9.8 (number out of 10)

Keep the residue numbers and values physically faithful to the data provided below.`;

    const dataPrompt = `Residue Selected ranges:
Residue i range: ${params.i_min} to ${params.i_max}
Residue j range: ${params.j_min} to ${params.j_max}
Residue distance |j - i| >= ${params.res_dist}
Analysis Mode: ${params.analysis_mode}

Top correlated residue-pairs to analyze (subset of data):
${JSON.stringify(correlationData, null, 2)}

Provide the JSON formatted output matches the exact schema requested. Do not wrap the JSON in markdown code blocks like \`\`\`json. Return pure JSON text.`;

    const response = await client.models.generateContent({
      model: modelToUse,
      contents: dataPrompt,
      config: {
        systemInstruction: systemPrompt,
        responseMimeType: "application/json",
      },
    });

    const text = response.text || "{}";
    res.json(JSON.parse(text));
  } catch (error: any) {
    console.error("Gemini API Error:", error);
    res.status(500).json({ error: error.message || "Failed to generate AI insights." });
  }
});

// Endpoint: AI-based residue pair specific explainer
app.post("/api/gemini/explain", async (req, res) => {
  try {
    const { residueI, residueJ, stats, model } = req.body;
    const client = getGemini();
    const modelToUse = model || "gemini-3.5-flash";

    const response = await client.models.generateContent({
      model: modelToUse,
      contents: `Provide an elegant molecular biophysics explanation of the interaction between residue ${residueI} and residue ${residueJ}.
Context Stats:
- Mean Correlation: ${stats.mean}
- Std Correlation: ${stats.std}
- Positive Probability: ${stats.posProb} (fraction of time they correlation is positive)
- Negative Probability: ${stats.negProb} (fraction of time correlation is negative)

Explain:
1. Is this correlation positive (concerted/in-phase motion) or negative (anti-correlated/out-of-phase motion)? What does that indicate structurally (e.g. hinge movement, domain breathing, rigid body movement, secondary structure lock)?
2. If the standard deviation is high, what does that indicate (e.g., conformational switching, intermittent encounters, flexible hinge region)?
3. What biochemical interactions (hydrogen binding, hydrophobic packing, salt bridges, or electrostatic interactions) are typical for these types of residues in macromolecular dynamics?
Provide a structured, beautiful response with brief sections. Use bullet points and professional terms. Keep it to around 150 words.`,
    });

    res.json({ explanation: response.text });
  } catch (error: any) {
    console.error("Gemini Explainer Error:", error);
    res.status(500).json({ error: error.message || "Failed to explain residue coupling." });
  }
});

// Configure Vite middleware or Static files
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://0.0.0.0:${PORT}`);
  });
}

if (process.env.NODE_ENV !== "production" || !process.env.VERCEL) {
  startServer();
}

export default app;
