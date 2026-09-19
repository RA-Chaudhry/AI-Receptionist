import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import asyncio
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)

class EmailService:
    @staticmethod
    def _send_smtp_sync(to_email: str, subject: str, body_text: str):
        if not settings.from_email:
            raise ValueError("FROM_EMAIL is not configured.")
        
        msg = MIMEMultipart("alternative")
        msg["From"] = settings.from_email
        msg["To"] = to_email
        msg["Subject"] = subject
        
        # Simple plain text or basic HTML format
        html_content = f"""
        <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
                    <h2 style="color: #4A90E2;">Logic Pulse - Pulse Concierge</h2>
                    <p>{body_text.replace(chr(10), '<br>')}</p>
                    <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
                    <p style="font-size: 0.8em; color: #777;">This is an automated response from Logic Pulse. Please do not reply directly to this email.</p>
                </div>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(body_text, "plain"))
        msg.attach(MIMEText(html_content, "html"))
        
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            if settings.smtp_use_tls:
                server.starttls()
            if settings.smtp_username and settings.smtp_password:
                server.login(settings.smtp_username, settings.smtp_password)
            server.send_message(msg)

    async def send_email(self, to_email: str, subject: str, body: str) -> bool:
        provider = settings.email_provider.lower()
        try:
            if provider == "smtp":
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._send_smtp_sync, to_email, subject, body)
                logger.info("Email sent successfully via SMTP")
                return True
                
            elif provider == "sendgrid":
                if not settings.sendgrid_api_key:
                    raise ValueError("SENDGRID_API_KEY is not set.")
                url = "https://api.sendgrid.com/v3/mail/send"
                headers = {
                    "Authorization": f"Bearer {settings.sendgrid_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "personalizations": [{"to": [{"email": to_email}]}],
                    "from": {"email": settings.from_email},
                    "subject": subject,
                    "content": [{"type": "text/plain", "value": body}]
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                logger.info("Email sent successfully via SendGrid")
                return True

            elif provider == "resend":
                if not settings.resend_api_key:
                    raise ValueError("RESEND_API_KEY is not set.")
                url = "https://api.resend.com/emails"
                headers = {
                    "Authorization": f"Bearer {settings.resend_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "from": settings.from_email,
                    "to": [to_email],
                    "subject": subject,
                    "text": body
                }
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                logger.info("Email sent successfully via Resend")
                return True
            else:
                raise ValueError(f"Unknown email provider: {provider}")
        except Exception as exc:
            logger.exception("Failed to send email")
            # Ensure system never crashes due to external email failure
            return False
