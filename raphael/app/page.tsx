import Header from "@/app/components/header";
import ChatSection from "./components/chat-section";
import Container from "./components/Container";
import ChatHistory from "./components/ChatHistory";

export default function Home() {
  return (
    <main className="flex flex-col items-center background-gradient ">

      <div className = "flex w-4/5 h-screen pt-10">
      <ChatHistory />
      <ChatSection />
      </div>
    </main>
  );
}
