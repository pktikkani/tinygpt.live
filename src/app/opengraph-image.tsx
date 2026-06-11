import { ImageResponse } from "next/og";

export const alt = "tinyGPT — a real GPT in your browser";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

const amber = "#f59e0b";
const green = "#22c55e";

function TinyRobot() {
  return (
    <svg width="360" height="360" viewBox="0 0 512 512">
      <g transform="translate(50 30)">
        <line x1="206" y1="92" x2="206" y2="44" stroke={amber} strokeWidth="14" />
        <circle cx="206" cy="30" r="22" fill={amber} />
        <rect x="66" y="92" width="280" height="216" rx="48" fill="#1e293b" stroke={amber} strokeWidth="12" />
        <circle cx="146" cy="182" r="26" fill={amber} />
        <circle cx="266" cy="182" r="26" fill={amber} />
        <path d="M 146 242 Q 206 280 266 242" stroke={amber} strokeWidth="14" fill="none" strokeLinecap="round" />
        <rect x="106" y="326" width="200" height="100" rx="32" fill="#1e293b" stroke={amber} strokeWidth="12" />
        <rect x="176" y="354" width="60" height="40" rx="10" fill={green} />
        <line x1="106" y1="354" x2="50" y2="394" stroke={amber} strokeWidth="14" strokeLinecap="round" />
        <line x1="306" y1="354" x2="362" y2="394" stroke={amber} strokeWidth="14" strokeLinecap="round" />
      </g>
    </svg>
  );
}

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          alignItems: "center",
          background: "#050505",
          padding: "0 80px",
          gap: 60,
        }}
      >
        <TinyRobot />
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div
            style={{
              fontSize: 104,
              fontWeight: 700,
              color: amber,
              letterSpacing: 2,
            }}
          >
            tinyGPT
          </div>
          <div style={{ fontSize: 36, color: "#e0e0e0", marginTop: 10 }}>
            A real GPT in your browser
          </div>
          <div style={{ fontSize: 26, color: "#999999", marginTop: 12 }}>
            Train it. Play with it. See inside it.
          </div>
          <div style={{ display: "flex", gap: 14, marginTop: 36 }}>
            {["~5,000 params", "no backend", "kids → engineers"].map(
              (chip, i) => (
                <div
                  key={chip}
                  style={{
                    border: "2px solid #1e1e1e",
                    background: "#0a0a0a",
                    borderRadius: 10,
                    padding: "10px 18px",
                    fontSize: 22,
                    color: i === 2 ? amber : green,
                  }}
                >
                  {chip}
                </div>
              )
            )}
          </div>
          <div style={{ fontSize: 24, color: "#666666", marginTop: 40 }}>
            tinygpt.live
          </div>
        </div>
      </div>
    ),
    size
  );
}
