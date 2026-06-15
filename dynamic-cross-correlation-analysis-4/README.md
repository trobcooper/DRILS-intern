# 🧬 Dynamic Cross-Correlation (DCC) Scientific Analyzer
### **High-Performance Full-Stack Biophysics Application**

An enterprise-grade, high-performance biophysical computational platform designed to ingest, process, and analyze dynamic cross-correlation (DCC) matrices from Molecular Dynamics (MD) trajectories. By combining rich interactive visualization canvases, robust time-series mathematical modeling, and secure server-side LLM coprocessing, this tool enables researchers to uncover hidden allosteric pathways, identify functional domain boundaries, and accelerate structure-based drug discovery.

---

## 🌟 Strategic Project Impact & Application

In computational biophysics, understanding how proteins move is as important as knowing their static structures. Proteins are dynamic machines; their function is governed by coordinated, long-range residues coupling over time.
This application solves key bottlenecks in traditional structural biology workflows:
* **Removes Visual Latency**: Scales matrix rendering to hundreds of residues via hardware-accelerated canvas components, eliminating DOM-node strain.
* **Integrates Spatial & Temporal Data**: Fuses dynamic 2D correlation heatmaps with continuous time-series plots to track correlation trajectories chronologically across simulation frames.
* **Democratizes AI Predictions**: Automates allosteric pathfinding and pocket-binding boundary classification through secure, contextual LLM prompts, mapping numerical correlative models to deep biochemical theories.

---

## 🏗️ SDE System Architecture

The platform is engineered using a robust, full-stack monorepo architecture designed for rapid development, type safety, and seamless container deployment.

```
       +--------------------------------------------------------+
       |                  React 19 SPA (Vite)                   |
       |  - Canvas Heatmap   - Recharts Trace   - Motion UI     |
       +----------------------------+---------------------------+
                                    |
                            (Secure JSON API)
                                    v
       +--------------------------------------------------------+
       |                 Express Node.js Server                 |
       |  - Port 3000 Ingress    - API Proxy Router             |
       |  - Static File Server   - Vite HMR Middleware (Dev)    |
       +----------------------------+---------------------------+
                                    |
                            (Google GenAI SDK)
                                    v
       +--------------------------------------------------------+
       |               Google Gemini Model API                 |
       |  - Pocket Analysis      - Allosteric Classification    |
       +--------------------------------------------------------+
```

### 1. Robust Full-Stack Runtime (`server.ts`)
The server runs on an Express backend configured to handle environment-agnostic execution:
* **Development Mode**: Mounts Vite directly as dynamic middleware via `createViteServer` running in `middlewareMode: true`. This allows hot-reloading and source mapping for both frontend assets and backend endpoints on a single combined socket.
* **Production Mode**: Serves pre-built, highly-optimized static client bundles from the `/dist` output folder. An catch-all route (`*`) ensures flawless single-page application (SPA) client-side routing.
* **Port Ingress Binding**: Hard-bound to port `3000` and host `0.0.0.0` to comply with automated container routing layers (e.g., Cloud Run, AWS ECS).

### 2. Secure Secret Isolation & Proxy Pipeline
To safeguard sensitive enterprise credentials:
* **Double-Blind Authentication**: All requests requiring LLM reasoning are tunneled through server-side proxy routes (`/api/*`).
* **Zero Client Leakage**: The client never receives, handles, or stores the `GEMINI_API_KEY`. It simply transmits lightweight, pre-formatted numerical correlation JSON models to the server. The server appends the secret environment-level key, executes the `@google/genai` token calculations, and sanitizes the returned Markdown string before responding to the client.

### 3. Build & Compiler Pipelines
The build stage compiles TypeScript flawlessly across both runtimes:
* **Vite Compilation**: Bundles the modern React front-end with tree-shaken assets, minifying chunks for immediate browser execution.
* **Esbuild Transpilation**: Compiles the Express background server into a standalone, single-file CommonJS module (`dist/server.cjs`) using `esbuild server.ts --bundle --platform=node --format=cjs`. This completely bypasses Node's strict ESM path validations for local imports, drastically reducing final build artifact size and startup latency.

---

## 🔬 Scientific & Mathematical Derivations

### 1. Dynamic Cross-Correlation (DCC) Formula
DCC measures the correlated spatial displacement vectors of protein backbones (typically $C_\alpha$ atoms) throughout a simulation. Given an MD trajectory with $F$ frames, the cross-correlation coefficient $C(i, j)$ between residue $i$ and residue $j$ is calculated as:

$$C(i, j) = \frac{\langle \Delta \mathbf{r}_i \cdot \Delta \mathbf{r}_j \rangle}{\sqrt{\langle \Delta \mathbf{r}_i^2 \rangle \langle \Delta \mathbf{r}_j^2 \rangle}}$$

Where:
* $\Delta \mathbf{r}_i = \mathbf{r}_i - \langle \mathbf{r}_i \rangle$ represents the displacement vector of residue $i$ from its mean position $\langle \mathbf{r}_i \rangle$.
* $\langle \cdot \rangle$ indicates the time average over all the frames $1, 2, \dots, F$.
* $C(i, j) \in [-1, +1]$:
  * $+1$: Fully correlated (moving in identical directions).
  * $-1$: Fully anti-correlated (moving in exact opposite directions).
  * $0$: Orthogonal/independent motions.

### 2. Time-Series Representation (The X-Axis)
The visualizer's **Time Series Dynamic Trajectory Trace** tracks structural movements chronological frame-by-frame:
* **X-Axis**: Represents the simulation snapshot index (e.g., *Frame 1*, *Frame 2*, ..., *Frame $F$*). These snapshots are chronologically sequential, spaced at equal time intervals (e.g., every 10 picoseconds) throughout the MD trajectory.
* **Y-Axis**: Outlines the exact correlation coefficient $C_t(i, j)$ at that discrete time slice $t$.

#### **Dynamic Trend Tracking**
* **Mean correlation ($\mu_{i, j}$)**:
  $$\mu_{i,j} = \frac{1}{F} \sum_{t=1}^{F} C_t(i, j)$$
* **Linear Slope ($m_{i,j}$)**:
  We fit a linear regression line across frames to evaluate structural relaxation or tightened coupling:
  $$m_{i, j} = \frac{F \sum (t \cdot C_t) - \sum t \sum C_t}{F \sum t^2 - (\sum t)^2}$$
  * A trend is labeled as **Increasing** ($m_{i,j} \ge 0.001$), indicating tightening dynamic correlation, or **Decreasing** ($m_{i,j} \le -0.001$), indicating independent relaxation.

---

## 💻 Tech Stack

- **Client Runtime**: React 19, TypeScript, Vite, Tailwind CSS, Motion (seamless transitions), and Recharts/D3 (charts/vector calculations).
- **Server Runtime**: Express, Node.js, TS-Node/TSX (development runner), Class-transformer.
- **Transpilation Pipeline**: Esbuild (production bundling), Vite CSS modules.
- **AI Integration**: Official Google `@google/genai` model client for secure server-side execution.

---

## 🛠️ Installation & Running Local

Follow these steps to set up, build, and deploy the application locally:

### 1. Install Dependencies
Ensure you have [Node.js](https://nodejs.org/) (v18+) and standard `npm` installed:
```bash
npm install
```

### 2. Environment Configuration
Create a `.env` file in the project's root directory and insert your Gemini credentials:
```env
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 3. Launch the Development Server
Starts the Express server with embedded Vite middleware and HMR enabled on port 3000:
```bash
npm run dev
```
Open **`http://localhost:3000`** in your browser.

### 4. Build and Run in Standalone Production
Pre-compiles the client assets, bundles the backend server into a portable CommonJS package (`dist/server.cjs`), and boots production ingress:
```bash
npm run build
npm start
```

---

## 🐙 Git Synchronization Workflows

Keep your changes synchronized with your GitHub repositories using either of these paths:

### Option A: Direct AI Studio Export (Automated)
1. Navigate to the top-right settings drawer (Gear icon) in the AI Studio environment.
2. Select **Export to GitHub**.
3. Authorize your target repository and create your remote database with a single click.

### Option B: Manual Command-Line Push
To push your downloaded source code manually to a blank remote repository:

```bash
# 1. Initialize local repository
git init

# 2. Stage system files (excluding build files covered by .gitignore)
git add .
git commit -m "feat: design enterprise full-stack DCC analyzer with secure proxy API rules"

# 3. Authenticate remote branch & deploy
git remote add origin https://github.com/YOUR_GITHUB_USER/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```
