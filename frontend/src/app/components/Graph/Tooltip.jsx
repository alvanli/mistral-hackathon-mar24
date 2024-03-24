import { useEffect } from "react";

export const Tooltip = ({ interactionData, clusters }) => {
  if (!interactionData) {
    return null;
  }
  
  return (
    <div
      style={{
        position: 'relative',
        left: interactionData.xPos + 400,
        top: interactionData.yPos + 200,
        backgroundColor: "#000000",
        padding: 5,
        width: Math.max(interactionData.name.length * 20, 70),
        textAlign: "center",
        fontSize: 15
      }}
    >
      <div
        style={{
          fontSize: 10,
          textAlign: "left"
        }}
      >
        {clusters[interactionData.label].summary !== "None" && `${clusters[interactionData.label].summary}: `}
      </div>
      {interactionData.name}
    </div>
  );
};
