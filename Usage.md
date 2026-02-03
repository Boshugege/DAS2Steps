# Usage

This is the exact command set used to run the sender/receiver locally.

## 1) Create venv and install dependency

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install nptdms
```

## 2) Start receiver (UDP, port 9000)

```bash
.\.venv\Scripts\python ReceiveJSON.py --port 9000 --max-messages 5 --timeout 5
```

## 3) Start sender (TDMS2JSONStream)

```bash
.\.venv\Scripts\python TDMS2JSONStream.py "C:\Users\boshu\Desktop\B202_1m_1000Hz_10m__UTC_20231115_045757.144.tdms" --port 9000 --sample-rate 1000 --max-samples 5
```

## 4) Full stream (no limits)

```bash
.\.venv\Scripts\python ReceiveJSON.py --port 9000 --timeout 10
.\.venv\Scripts\python TDMS2JSONStream.py "C:\Users\boshu\Desktop\B202_1m_1000Hz_10m__UTC_20231115_045757.144.tdms" --port 9000 --sample-rate 1000
```

## JSON output format

Each line is one JSON object (newline-delimited JSON). The fields are:

```json
{
  "total_channels": 448,
  "timestamp": 0.0,
  "sample_rate": 1000,
  "sample_count": 1,
  "signals": [-10008.0, -7146.0, "..."]
}
```

Notes:
- `total_channels`: number of channels included in `signals`.
- `timestamp`: seconds since the first sample for this packet.
- `sample_rate`: Hz (used for timestamps).
- `sample_count`: always 1 (one sample per packet).
- `signals`: one row of `[ch0, ch1, ...]` at one sample index.
- If `--start-time-iso` is provided, an extra field appears:
  - `timestamp_iso`: ISO-8601 timestamp for the sample.

## Unity API (UDP receiver)

This stream is newline-delimited JSON over UDP on the configured port.

Packet schema:

```json
{
  "total_channels": 448,
  "timestamp": 0.001,
  "sample_rate": 1000,
  "sample_count": 1,
  "signals": [123.4, 125.6, "..."],
  "timestamp_iso": "2025-01-01T00:00:00+00:00"
}
```

Unity receiver example (C#):

```csharp
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

[Serializable]
public class TdmsPacket
{
    public int total_channels;
    public double timestamp;
    public double sample_rate;
    public int sample_count;
    public double[] signals;
    public string timestamp_iso;
}

public class TdmsUdpReceiver : MonoBehaviour
{
    public int port = 9000;
    private UdpClient _udp;
    private IPEndPoint _ep;

    void Start()
    {
        _udp = new UdpClient(port);
        _ep = new IPEndPoint(IPAddress.Any, port);
    }

    void Update()
    {
        while (_udp.Available > 0)
        {
            byte[] data = _udp.Receive(ref _ep);
            string json = Encoding.UTF8.GetString(data).Trim();
            TdmsPacket packet = JsonUtility.FromJson<TdmsPacket>(json);
            // Use packet.signals here.
        }
    }

    void OnDestroy()
    {
        _udp?.Close();
    }
}
```

Notes for Unity:
- Use UDP because packets are independent; order is by arrival.
- Each UDP datagram contains exactly one JSON object.

C:/Users/boshu/Desktop/srt/DAS2Steps/.venv/Scripts/python.exe c:/Users/boshu/Desktop/srt/DAS2Steps/TDMS2JSONStream.py "C:\Users\boshu\Desktop\B202_1m_1000Hz_10m__UTC_20231115_071033.961.tdms" --sample-rate 1000 --wavelet-type sym4 --wavelet-level 4 --denoise-wavelet db4 --denoise-mode hard

python JSONStreamPlot.py --protocol udp --show-events
