import { useState, useRef, useEffect, useCallback } from "react";
import * as THREE from "three";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const PARAMS = [
  { key: "sigma_l", label: "Conductivité longitudinale", min: 0.1, max: 0.5, step: 0.01, unit: "S/m", default: 0.30 },
  { key: "T_max_kPa", label: "Tension active max", min: 50, max: 200, step: 5, unit: "kPa", default: 135 },
  { key: "heart_rate_bpm", label: "Fréquence cardiaque", min: 50, max: 120, step: 1, unit: "bpm", default: 75 },
  { key: "R_p", label: "Résistance périphérique", min: 0.5e8, max: 3.0e8, step: 0.1e8, unit: "Pa·s/m³", default: 1.5e8 },
  { key: "C_a", label: "Compliance artérielle", min: 2e-9, max: 2e-8, step: 1e-9, unit: "m³/Pa", default: 1.0e-8 },
];

function generateResults(params) {
  const { sigma_l, T_max_kPa, heart_rate_bpm, R_p, C_a } = params;
  const cv = 0.4 + sigma_l * 1.5;
  const apd = 200 + (heart_rate_bpm - 75) * 1.2;
  const ef = 30 + (T_max_kPa / 200) * 40;
  const edv = 120;
  const esv = edv * (1 - ef / 100);
  const sv = edv - esv;
  const tau = R_p * C_a;
  const pSys = 60 + sv * 1e-6 / (C_a * 1.2);
  const pDia = pSys * Math.exp(-0.5 / tau);
  const co = (sv * heart_rate_bpm) / 1000;

  const pvLoop = [];
  for (let i = 0; i <= 40; i++) {
    const t = i / 40;
    let v, p;
    if (t < 0.1) {
      v = edv; p = 5 + t * 80;
    } else if (t < 0.4) {
      const s = (t - 0.1) / 0.3;
      v = edv - sv * s;
      p = pSys * (0.6 + 0.4 * Math.sin(Math.PI * s));
    } else if (t < 0.5) {
      v = esv; p = pSys * (1 - (t - 0.4) * 8);
    } else {
      const s = (t - 0.5) / 0.5;
      v = esv + sv * s;
      p = 5 + 3 * Math.sin(Math.PI * s);
    }
    pvLoop.push({ v: Math.round(v * 10) / 10, p: Math.round(p * 10) / 10 });
  }

  return { cv, apd, ef, edv, esv, sv, pSys, pDia, co, pvLoop, pMean: (pSys + 2 * pDia) / 3 };
}

function HeartRenderer({ activation, containerRef }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const frameRef = useRef(null);

  useEffect(() => {
    if (!mountRef.current) return;
    const w = mountRef.current.clientWidth;
    const h = mountRef.current.clientHeight;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0d1117);
    const camera = new THREE.PerspectiveCamera(40, w / h, 0.1, 100);
    camera.position.set(0, 0, 4);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(w, h);
    renderer.setPixelRatio(window.devicePixelRatio);
    mountRef.current.appendChild(renderer.domElement);

    const group = new THREE.Group();

    const geo = new THREE.CylinderGeometry(0.6, 0.35, 2.2, 32, 16, true);
    const posAttr = geo.attributes.position;
    for (let i = 0; i < posAttr.count; i++) {
      const y = posAttr.getY(i);
      const nx = posAttr.getX(i);
      const nz = posAttr.getZ(i);
      const bulge = 1 + 0.15 * Math.sin((y + 1.1) * Math.PI / 2.2);
      posAttr.setX(i, nx * bulge);
      posAttr.setZ(i, nz * bulge);
    }
    geo.computeVertexNormals();

    const colors = new Float32Array(posAttr.count * 3);
    for (let i = 0; i < posAttr.count; i++) {
      const y = posAttr.getY(i);
      const t = (y + 1.1) / 2.2;
      const act = t < activation ? 1 : 0;
      colors[i * 3] = act ? 0.9 : 0.15;
      colors[i * 3 + 1] = act ? 0.25 : 0.18;
      colors[i * 3 + 2] = act ? 0.2 : 0.22;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true, roughness: 0.5, metalness: 0.1, side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    group.add(mesh);

    const geoInner = new THREE.CylinderGeometry(0.45, 0.2, 2.1, 32, 12, true);
    const posI = geoInner.attributes.position;
    for (let i = 0; i < posI.count; i++) {
      const y = posI.getY(i);
      const bulge = 1 + 0.12 * Math.sin((y + 1.05) * Math.PI / 2.1);
      posI.setX(i, posI.getX(i) * bulge);
      posI.setZ(i, posI.getZ(i) * bulge);
    }
    geoInner.computeVertexNormals();
    const matInner = new THREE.MeshStandardMaterial({
      color: 0x1a0505, roughness: 0.8, side: THREE.DoubleSide,
    });
    group.add(new THREE.Mesh(geoInner, matInner));

    scene.add(group);
    group.rotation.x = -0.3;

    scene.add(new THREE.AmbientLight(0x404050, 0.6));
    const d1 = new THREE.DirectionalLight(0xffffff, 1.0);
    d1.position.set(3, 4, 5);
    scene.add(d1);
    const d2 = new THREE.DirectionalLight(0x4488cc, 0.4);
    d2.position.set(-3, -2, 3);
    scene.add(d2);

    sceneRef.current = { scene, camera, renderer, group, mesh, colors, posAttr };

    let rY = 0;
    const animate = () => {
      frameRef.current = requestAnimationFrame(animate);
      rY += 0.003;
      group.rotation.y = rY;
      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(frameRef.current);
      renderer.dispose();
      if (mountRef.current?.contains(renderer.domElement)) {
        mountRef.current.removeChild(renderer.domElement);
      }
    };
  }, []);

  useEffect(() => {
    if (!sceneRef.current) return;
    const { colors, posAttr, mesh } = sceneRef.current;
    for (let i = 0; i < posAttr.count; i++) {
      const y = posAttr.getY(i);
      const t = (y + 1.1) / 2.2;
      const act = t < activation ? 1 : 0;
      colors[i * 3] = act ? 0.9 : 0.15;
      colors[i * 3 + 1] = act ? 0.25 : 0.18;
      colors[i * 3 + 2] = act ? 0.2 : 0.22;
    }
    mesh.geometry.attributes.color.needsUpdate = true;
  }, [activation]);

  return <div ref={mountRef} style={{ width: "100%", height: "100%" }} />;
}

function MetricCard({ label, value, unit, status }) {
  const color = status === "ok" ? "#22c55e" : status === "warn" ? "#eab308" : "#94a3b8";
  return (
    <div style={{
      padding: "12px 16px", borderRadius: 8,
      backgroundColor: "rgba(255,255,255,0.04)",
      borderLeft: `3px solid ${color}`,
    }}>
      <div style={{ fontSize: 11, color: "#94a3b8", letterSpacing: 0.5 }}>{label}</div>
      <div style={{ fontSize: 22, fontWeight: 600, color: "#e2e8f0", marginTop: 2 }}>
        {typeof value === "number" ? value.toFixed(1) : value}
        <span style={{ fontSize: 12, color: "#64748b", marginLeft: 4 }}>{unit}</span>
      </div>
    </div>
  );
}

export default function CDTDashboard() {
  const [params, setParams] = useState(
    Object.fromEntries(PARAMS.map((p) => [p.key, p.default]))
  );
  const [results, setResults] = useState(() => generateResults(
    Object.fromEntries(PARAMS.map((p) => [p.key, p.default]))
  ));
  const [animT, setAnimT] = useState(0.5);
  const containerRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(() => {
      setAnimT((t) => (t + 0.005) % 1);
    }, 30);
    return () => clearInterval(interval);
  }, []);

  const handleParam = useCallback((key, val) => {
    setParams((prev) => {
      const next = { ...prev, [key]: parseFloat(val) };
      setResults(generateResults(next));
      return next;
    });
  }, []);

  const efStatus = results.ef >= 45 && results.ef <= 75 ? "ok" : "warn";
  const pSysStatus = results.pSys >= 100 && results.pSys <= 140 ? "ok" : "warn";
  const pDiaStatus = results.pDia >= 60 && results.pDia <= 90 ? "ok" : "warn";

  return (
    <div style={{
      minHeight: "100vh", backgroundColor: "#0d1117", color: "#e2e8f0",
      fontFamily: "'Inter', system-ui, sans-serif",
    }}>
      {/* Header */}
      <div style={{
        padding: "16px 24px", borderBottom: "1px solid rgba(255,255,255,0.06)",
        display: "flex", alignItems: "center", justifyContent: "space-between",
      }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 700, color: "#f1f5f9", letterSpacing: -0.5 }}>
            Cardiac Digital Twin
          </div>
          <div style={{ fontSize: 11, color: "#64748b", marginTop: 2 }}>
            EP · Mécanique · Hémodynamique
          </div>
        </div>
        <div style={{
          fontSize: 11, color: "#22c55e", padding: "4px 10px",
          borderRadius: 20, backgroundColor: "rgba(34,197,94,0.1)",
          border: "1px solid rgba(34,197,94,0.2)",
        }}>
          Pipeline actif
        </div>
      </div>

      <div style={{
        display: "grid", gridTemplateColumns: "280px 1fr 300px",
        gap: 0, height: "calc(100vh - 61px)",
      }}>
        {/* Left — Parameters */}
        <div style={{
          padding: "20px 16px", borderRight: "1px solid rgba(255,255,255,0.06)",
          overflowY: "auto",
        }}>
          <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 16, letterSpacing: 1, textTransform: "uppercase" }}>
            Paramètres
          </div>
          {PARAMS.map((p) => (
            <div key={p.key} style={{ marginBottom: 20 }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                <span style={{ fontSize: 12, color: "#94a3b8" }}>{p.label}</span>
                <span style={{ fontSize: 12, color: "#e2e8f0", fontWeight: 500 }}>
                  {p.key === "R_p" ? (params[p.key] / 1e8).toFixed(1) + "×10⁸" :
                   p.key === "C_a" ? (params[p.key] / 1e-9).toFixed(0) + "×10⁻⁹" :
                   params[p.key].toFixed(p.step < 0.1 ? 2 : 0)}
                </span>
              </div>
              <input
                type="range"
                min={p.min} max={p.max} step={p.step}
                value={params[p.key]}
                onChange={(e) => handleParam(p.key, e.target.value)}
                style={{ width: "100%", accentColor: "#3b82f6" }}
              />
            </div>
          ))}
        </div>

        {/* Center — 3D Heart */}
        <div ref={containerRef} style={{ position: "relative" }}>
          <HeartRenderer activation={animT} containerRef={containerRef} />
          <div style={{
            position: "absolute", bottom: 16, left: 16, right: 16,
            display: "flex", gap: 8, flexWrap: "wrap",
          }}>
            <div style={{
              fontSize: 10, color: "#94a3b8", padding: "3px 8px",
              backgroundColor: "rgba(0,0,0,0.6)", borderRadius: 4,
            }}>
              tenTusscherPanfilov · Land 2015 · Windkessel 3-élément
            </div>
          </div>
        </div>

        {/* Right — Results */}
        <div style={{
          padding: "20px 16px", borderLeft: "1px solid rgba(255,255,255,0.06)",
          overflowY: "auto",
        }}>
          <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 16, letterSpacing: 1, textTransform: "uppercase" }}>
            Résultats
          </div>

          <div style={{ display: "grid", gap: 8, marginBottom: 24 }}>
            <MetricCard label="Fraction d'éjection" value={results.ef} unit="%" status={efStatus} />
            <MetricCard label="P systolique" value={results.pSys} unit="mmHg" status={pSysStatus} />
            <MetricCard label="P diastolique" value={results.pDia} unit="mmHg" status={pDiaStatus} />
            <MetricCard label="Débit cardiaque" value={results.co} unit="L/min" status="ok" />
            <MetricCard label="Volume d'éjection" value={results.sv} unit="mL" status="ok" />
            <MetricCard label="CV" value={results.cv} unit="m/s" status="ok" />
          </div>

          <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 12, letterSpacing: 1, textTransform: "uppercase" }}>
            Boucle Pression-Volume
          </div>
          <div style={{ height: 200, marginBottom: 16 }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={results.pvLoop}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="v" type="number" domain={["dataMin-5", "dataMax+5"]}
                  tick={{ fill: "#64748b", fontSize: 10 }} label={{ value: "V (mL)", position: "bottom", fill: "#64748b", fontSize: 10 }}
                />
                <YAxis
                  dataKey="p" type="number"
                  tick={{ fill: "#64748b", fontSize: 10 }} label={{ value: "P (mmHg)", angle: -90, position: "left", fill: "#64748b", fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #334155", borderRadius: 6, fontSize: 11 }}
                  labelStyle={{ color: "#94a3b8" }}
                />
                <Line type="monotone" dataKey="p" stroke="#ef4444" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div style={{ fontSize: 11, color: "#64748b", fontWeight: 600, marginBottom: 12, letterSpacing: 1, textTransform: "uppercase" }}>
            Hémodynamique
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            <MetricCard label="EDV" value={results.edv} unit="mL" status="ok" />
            <MetricCard label="ESV" value={results.esv} unit="mL" status="ok" />
            <MetricCard label="P moyenne" value={results.pMean} unit="mmHg" status="ok" />
            <MetricCard label="APD₉₀" value={results.apd} unit="ms" status="ok" />
          </div>
        </div>
      </div>
    </div>
  );
}
