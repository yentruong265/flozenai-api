import dotenv from "dotenv";
dotenv.config();

import express from "express";
import cors from "cors";

const app = express();

app.use(cors({
  origin: [
    "https://app.flozenai.com",
    "http://localhost:3000",
    "http://127.0.0.1:3000"
  ]
}));

app.use(express.json());

const PORT = process.env.PORT || 3001;
const ADMIN_EMAIL = process.env.ADMIN_EMAIL;
const ADMIN_PASSWORD = process.env.ADMIN_PASSWORD;
const RUNPOD_API_URL = process.env.RUNPOD_API_URL;
const RUNPOD_API_KEY = process.env.RUNPOD_API_KEY;

function requireAdmin(req, res, next) {
  const { email, password } = req.body || {};
  if (email !== ADMIN_EMAIL || password !== ADMIN_PASSWORD) {
    return res.status(401).json({ ok: false, error: "Unauthorized" });
  }
  next();
}

app.get("/api/health", (req, res) => {
  res.json({ ok: true, message: "FlozenAI API is running" });
});

app.post("/api/login", (req, res) => {
  const { email, password } = req.body || {};

  if (email !== ADMIN_EMAIL || password !== ADMIN_PASSWORD) {
    return res.status(401).json({ ok: false, error: "Unauthorized" });
  }

  return res.json({
    ok: true,
    user: {
      email,
      role: "admin"
    }
  });
});

app.post("/api/generate", requireAdmin, async (req, res) => {
  try {
    const { payload } = req.body || {};

    if (!RUNPOD_API_URL || !RUNPOD_API_KEY || RUNPOD_API_URL === "dummy" || RUNPOD_API_KEY === "dummy") {
      return res.status(500).json({
        ok: false,
        error: "Runpod config not ready yet"
      });
    }

    const rpRes = await fetch(RUNPOD_API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${RUNPOD_API_KEY}`
      },
      body: JSON.stringify({
        input: payload
      })
    });

    const data = await rpRes.json();

    return res.json({
      ok: true,
      runpod: data
    });
  } catch (e) {
    return res.status(500).json({
      ok: false,
      error: String(e)
    });
  }
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`FlozenAI API running on port ${PORT}`);
});