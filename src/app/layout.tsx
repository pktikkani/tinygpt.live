import type { Metadata, Viewport } from "next";
import { Geist_Mono } from "next/font/google";
import "./globals.css";
import PwaRegister from "@/components/PwaRegister";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// Resolves to the custom domain once one is attached to the Vercel project;
// until then social cards use the canonical *.vercel.app production URL.
const siteUrl = process.env.VERCEL_PROJECT_PRODUCTION_URL
  ? `https://${process.env.VERCEL_PROJECT_PRODUCTION_URL}`
  : "https://tinygpt.live";

export const metadata: Metadata = {
  metadataBase: new URL(siteUrl),
  title: "tinyGPT — Interactive GPT Visualizer",
  description:
    "Watch tokens flow through a real GPT in your browser. Inspect attention heads, control temperature, step through training.",
  applicationName: "tinyGPT",
  openGraph: {
    title: "tinyGPT — A real GPT in your browser",
    description:
      "Train a real transformer in your browser, play guessing games against it, and see attention with your own eyes. For kids and engineers alike.",
    url: siteUrl,
    siteName: "tinyGPT",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "tinyGPT — A real GPT in your browser",
    description:
      "Train a real transformer in your browser, play guessing games against it, and see attention with your own eyes.",
  },
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "tinyGPT",
  },
  icons: {
    icon: [
      { url: "/icon-192.png", sizes: "192x192", type: "image/png" },
      { url: "/icon.svg", type: "image/svg+xml" },
    ],
    apple: "/apple-touch-icon.png",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#fafafa" },
    { media: "(prefers-color-scheme: dark)", color: "#050505" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${geistMono.variable} antialiased`}>
        {children}
        <PwaRegister />
      </body>
    </html>
  );
}
