
/* PA-LAW chat keyboard shortcuts
   Enter       = Send message
   Shift+Enter = Insert a new line
*/

(function () {
    "use strict";

    document.addEventListener("keydown", function (event) {
        const target = event.target;

        // Only apply to the PA-LAW chat textarea
        if (!target || target.id !== "chat-input") {
            return;
        }

        // Enter sends the message
        // Shift+Enter inserts a new line
        if (
            event.key === "Enter" &&
            !event.shiftKey &&
            !event.isComposing
        ) {
            event.preventDefault();

            // Prevent repeated submissions while holding Enter
            if (event.repeat) {
                return;
            }

            // Find and click the existing Send button
            const sendButton = document.getElementById("send-chat");

            if (sendButton && !sendButton.disabled) {
                sendButton.click();
            }
        }
    }, true);
})();
