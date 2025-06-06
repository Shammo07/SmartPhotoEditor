from django.db import models

class Photo(models.Model):
    original = models.ImageField(upload_to='uploads/')
    edited = models.ImageField(upload_to='edited/', null=True, blank=True)
    preview = models.ImageField(upload_to='previews/', null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Photo {self.id}"