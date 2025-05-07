using System;
using System.Net.Http;
using System.Text;
using System.IO;
using System.Text.Json;
using System.Threading.Tasks;

// Convert image to base64
string query = "I am at Server Room 1, what should I do?";
string imagePath = "test_images/im4.png"; // Update this path

if (!File.Exists(imagePath))
{
    Console.WriteLine($"Image not found at: {imagePath}");
    return;
}

string base64Image = Convert.ToBase64String(File.ReadAllBytes(imagePath));

var requestBody = new
{
    query = query,
    image_data = base64Image
};

string json = JsonSerializer.Serialize(requestBody);
var content = new StringContent(json, Encoding.UTF8, "application/json");

HttpClient client = new HttpClient();

try
{
    HttpResponseMessage response = await client.PostAsync("http://localhost:5005/ask", content);
    string responseBody = await response.Content.ReadAsStringAsync();

    Console.WriteLine($"Status: {response.StatusCode}");
    Console.WriteLine("Response:");
    Console.WriteLine(responseBody);
}
catch (Exception ex)
{
    Console.WriteLine($"Error: {ex.Message}");
}
