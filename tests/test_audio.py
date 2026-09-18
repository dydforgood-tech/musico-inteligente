"""Testes automatizados da camada de áudio (v0.1-A)."""

import os
import tempfile
import unittest
import numpy as np
import soundfile as sf

from app.audio.audio_loader import FileAudioSource
from app.audio.audio_player import AudioPlayer, PlaybackState
from app.utils.audio_generator import generate_test_song, generate_tone


class TestAudioLayer(unittest.TestCase):
    """Validação rigorosa do motor de áudio para arquivos WAV/MP3, envelopes e seek."""

    @classmethod
    def setUpClass(cls):
        """Gera arquivos de teste temporários para a suíte."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.wav_file = os.path.join(cls.temp_dir, "test_synth_120bpm.wav")
        generate_test_song(cls.wav_file, bpm=120.0)

        # Gerar arquivo mono de silêncio para testar robustez a divisão por zero
        cls.silence_file = os.path.join(cls.temp_dir, "silence.wav")
        sf.write(cls.silence_file, np.zeros((44100, 1), dtype=np.float32), 44100)

        # Gerar arquivo MP3 de teste
        cls.mp3_file = os.path.join(cls.temp_dir, "test_synth.mp3")
        tone = generate_tone(440.0, 1.5, sr=44100)
        sf.write(cls.mp3_file, tone, 44100, format="MP3")

    @classmethod
    def tearDownClass(cls):
        """Remove arquivos temporários criados."""
        for path in [cls.wav_file, cls.silence_file, cls.mp3_file]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
        if os.path.exists(cls.temp_dir):
            try:
                os.rmdir(cls.temp_dir)
            except Exception:
                pass

    def test_file_audio_source_mp3(self):
        """Valida carregamento de arquivos MP3 nativamente."""
        source = FileAudioSource(self.mp3_file)
        self.assertEqual(source.get_sample_rate(), 44100)
        self.assertTrue(source.is_active())
        self.assertAlmostEqual(source.get_duration(), 1.5, places=1)
        chunk = source.read_chunk(1024)
        self.assertEqual(chunk.shape, (1024,))
        source.close()

    def test_file_audio_source_metadata(self):
        """Valida se taxa de amostragem, canais e duração são extraídos com precisão."""
        source = FileAudioSource(self.wav_file)
        self.assertEqual(source.get_sample_rate(), 44100)
        self.assertEqual(source.channels, 2)
        self.assertTrue(source.is_active())
        self.assertGreater(source.get_duration(), 15.0)
        self.assertLess(source.get_duration(), 17.0)
        source.close()

    def test_audio_read_chunks(self):
        """Valida leitura sequencial de chunks em mono e multicanal."""
        source = FileAudioSource(self.wav_file)
        chunk_mono = source.read_chunk(1024)
        self.assertEqual(chunk_mono.shape, (1024,))
        self.assertEqual(chunk_mono.dtype, np.float32)
        self.assertTrue(np.all(chunk_mono >= -1.0) and np.all(chunk_mono <= 1.0))

        chunk_playback = source.read_playback_chunk(512)
        self.assertEqual(chunk_playback.shape, (512, 2))
        self.assertEqual(chunk_playback.dtype, np.float32)
        source.close()

    def test_audio_seek(self):
        """Valida movimentação do cursor de leitura (seek)."""
        source = FileAudioSource(self.wav_file)
        source.seek(5.0)
        self.assertAlmostEqual(source.get_position(), 5.0, places=2)

        # Seek além da duração deve limitar ao final seguro
        source.seek(999.0)
        self.assertAlmostEqual(source.get_position(), source.get_duration(), places=2)

        # Seek negativo deve limitar a 0.0
        source.seek(-10.0)
        self.assertEqual(source.get_position(), 0.0)
        source.close()

    def test_waveform_envelope(self):
        """Valida cálculo de picos mínimo/máximo para a visualização gráfica."""
        source = FileAudioSource(self.wav_file)
        num_points = 300
        mins, maxs = source.get_waveform_envelope(num_points)
        self.assertEqual(len(mins), num_points)
        self.assertEqual(len(maxs), num_points)
        self.assertTrue(np.all(mins <= maxs))
        source.close()

    def test_silence_handling(self):
        """Valida que áudio com amplitude zero não gera erros de NaN ou ZeroDivisionError."""
        source = FileAudioSource(self.silence_file)
        self.assertEqual(source.get_duration(), 1.0)
        mins, maxs = source.get_waveform_envelope(100)
        self.assertEqual(np.max(np.abs(mins)), 0.0)
        self.assertEqual(np.max(np.abs(maxs)), 0.0)
        source.close()

    def test_audio_player_state(self):
        """Valida máquina de estados e transporte do AudioPlayer."""
        player = AudioPlayer()
        self.assertEqual(player.state, PlaybackState.STOPPED)

        source = FileAudioSource(self.wav_file)
        player.load_source(source)
        self.assertEqual(player.state, PlaybackState.STOPPED)

        # Seek no player
        player.seek(2.5)
        self.assertAlmostEqual(player.get_position(), 2.5, places=1)

        player.stop()
        self.assertEqual(player.get_position(), 0.0)
        player.close()


if __name__ == "__main__":
    unittest.main()
