import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt


class AssociativeRecallExperiment:
    """Evaluate associative recall using key-value sequences."""

    def __init__(self, model, vocab_size=1000, max_pairs=50):
        self.model = model
        self.vocab_size = vocab_size
        self.max_pairs = max_pairs

    def generate_task(self, num_pairs, query_position='random'):
        """Create a single associative recall task instance."""
        keys = torch.randint(0, self.vocab_size // 2, (num_pairs,))
        values = torch.randint(self.vocab_size // 2, self.vocab_size, (num_pairs,))

        sequence = []
        for key, value in zip(keys, values):
            sequence.extend([key.item(), value.item()])

        if query_position == 'random':
            query_idx = torch.randint(0, num_pairs, (1,)).item()
        else:
            query_idx = query_position

        query_key = keys[query_idx].item()
        target_value = values[query_idx].item()

        sequence.extend([query_key, -100])

        input_ids = torch.tensor(sequence[:-1]).unsqueeze(0)
        target_ids = torch.tensor(sequence[1:]).unsqueeze(0)
        target_ids[0, :-1] = -100

        return input_ids, target_ids, target_value

    def evaluate(self, num_trials=100, num_pairs_range=(5, 50)):
        """Run evaluation across a range of pair counts."""
        bins = range(5, self.max_pairs + 1, 5)
        results = {n: {'correct': 0, 'total': 0} for n in bins}

        self.model.eval()
        with torch.no_grad():
            for _ in tqdm(range(num_trials)):
                num_pairs = torch.randint(num_pairs_range[0], num_pairs_range[1] + 1, (1,)).item()
                input_ids, target_ids, target_value = self.generate_task(num_pairs)

                input_ids = input_ids.to(self.model.device)
                logits = self.model(input_ids)
                prediction = logits[0, -1].argmax().item()

                bracket = (num_pairs // 5) * 5
                if bracket in results:
                    results[bracket]['total'] += 1
                    if prediction == target_value:
                        results[bracket]['correct'] += 1

        accuracies = {}
        for n, stats in results.items():
            accuracies[n] = stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0

        plt.figure(figsize=(10, 6))
        buckets = sorted(accuracies.keys())
        plt.plot(buckets, [accuracies[b] for b in buckets], 'o-', linewidth=2, markersize=8)
        plt.xlabel('Number of Key-Value Pairs')
        plt.ylabel('Accuracy')
        plt.title('Associative Recall Performance')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1.05])
        plt.savefig('associative_recall_results.png', dpi=150)

        return accuracies


class ContinualLearningExperiment:
    """Sequentially train on tasks and measure forgetting."""

    def __init__(self, model, num_tasks=5):
        self.model = model
        self.num_tasks = num_tasks
        self.tasks = self._generate_tasks()

    def _generate_tasks(self):
        tasks = []
        for task_id in range(self.num_tasks):
            start_token = task_id * 100
            end_token = (task_id + 1) * 100
            tasks.append({
                'id': task_id,
                'name': f'Copy tokens [{start_token}-{end_token})',
                'token_range': (start_token, end_token),
            })
        return tasks

    def train_task(self, task_id, num_steps=100):
        task = self.tasks[task_id]
        start, end = task['token_range']

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        for _ in range(num_steps):
            seq_len = 10
            input_tokens = torch.randint(start, end, (1, seq_len))
            target_tokens = input_tokens.clone()

            input_tokens = input_tokens.to(self.model.device)
            target_tokens = target_tokens.to(self.model.device)

            logits = self.model(input_tokens, update_memory=True)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target_tokens.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def evaluate_all_tasks(self):
        self.model.eval()
        results = {}

        with torch.no_grad():
            for task in self.tasks:
                task_id = task['id']
                start, end = task['token_range']

                correct = 0
                total = 0

                for _ in range(20):
                    seq_len = 10
                    input_tokens = torch.randint(start, end, (1, seq_len))
                    target_tokens = input_tokens.clone()

                    input_tokens = input_tokens.to(self.model.device)

                    logits = self.model(input_tokens, update_memory=False)
                    predictions = logits.argmax(dim=-1)

                    correct += (predictions == target_tokens.to(self.model.device)).sum().item()
                    total += seq_len

                results[task_id] = correct / total if total > 0 else 0.0

        return results

    def run_continual_learning(self):
        history = []

        for task_id in range(self.num_tasks):
            print(f"\nTraining Task {task_id}...")
            self.train_task(task_id)

            accuracies = self.evaluate_all_tasks()
            history.append(accuracies)

            print(f"After training task {task_id}:")
            for tid, acc in accuracies.items():
                if tid <= task_id:
                    print(f"  Task {tid}: {acc:.2%}")

        self._plot_forgetting(history)
        return history

    def _plot_forgetting(self, history):
        num_evaluated = len(history)
        matrix = np.zeros((num_evaluated, self.num_tasks))

        for i, accuracies in enumerate(history):
            for task_id, acc in accuracies.items():
                if task_id < self.num_tasks:
                    matrix[i, task_id] = acc

        plt.figure(figsize=(10, 8))
        plt.imshow(matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(label='Accuracy')
        plt.xlabel('After Training Task N')
        plt.ylabel('Task ID')
        plt.title('Continual Learning: Accuracy Matrix\n(Diagonal = current task, Off-diagonal = retention)')
        plt.xticks(range(num_evaluated), [f'Task {i}' for i in range(num_evaluated)])
        plt.yticks(range(self.num_tasks), [f'Task {i}' for i in range(self.num_tasks)])
        plt.tight_layout()
        plt.savefig('continual_learning_results.png', dpi=150)

        forgetting = []
        for task_id in range(self.num_tasks - 1):
            peak_acc = matrix[task_id, task_id]
            final_acc = matrix[-1, task_id]
            forgetting.append(peak_acc - final_acc)

        avg_forgetting = np.mean(forgetting) if forgetting else 0.0
        print(f"\nAverage forgetting: {avg_forgetting:.2%}")
