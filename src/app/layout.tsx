import type { Metadata, Viewport } from "next";
import { Geist_Mono } from "next/font/google";
import "./globals.css";
import PwaRegister from "@/components/PwaRegister";

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL("https://tinygpt.live"),
  title: "tinyGPT — Interactive GPT Visualizer",
  description:
    "Watch tokens flow through a real GPT in your browser. Inspect attention heads, control temperature, step through training.",
  applicationName: "tinyGPT",
  openGraph: {
    title: "tinyGPT — A real GPT in your browser",
    description:
      "Train a real transformer in your browser, play guessing games against it, and see attention with your own eyes. For kids and engineers alike.",
    url: "https://tinygpt.live",
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
