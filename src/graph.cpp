#include "../include/graph.hpp"

#include <deal.II/base/exceptions.h>
#include <fstream>

#include<stack>

void
Graph::add_point(double x, double y, double z) {
  points.push_back(Point<3>(x, y,z));
  adiacency.push_back(std::vector<int>());
}

void
Graph::add_point(Point<3> point) {
  points.push_back(point);
  adiacency.push_back(std::vector<int>());
}

void
Graph::add_cell(int node1, int node2) {
  CellData<1> cell;

  cell.vertices[0] = node1;
  cell.vertices[1] = node2;

  adiacency[node1].push_back(cells.size());
  adiacency[node2].push_back(cells.size());
  
  cells.push_back(cell);
} 

void 
Graph::add_big_cell(int node1, int node2, int cell_start, int cell_end, int n_of_cells) {
  BigCell big_cell;

  big_cell.node1 = node1;
  big_cell.node2 = node2;
  big_cell.cell_start = cell_start;
  big_cell.cell_end = cell_end;
  big_cell.n_of_cells = n_of_cells;

  big_cells.push_back(big_cell);
}


Triangulation<1, 3>
Graph::create_graph_triangulation() {
  Triangulation<1, 3> result;

  result.create_triangulation (points, cells, SubCellData());

  return result;
}

int 
Graph::get_number_of_points() {
  return points.size();
}

int
Graph::get_number_of_cells() {
  return cells.size();
}

Graph
Graph::get_coarser_graph(std::map<int, std::vector<int>> & coarse_to_fine_vertex_map,
  std::map<int, int> & coarse_to_fine_cell_map,
  double coarsening_percentage,
  const std::vector<double> & resistances) {

  Graph result;

  std::vector<double> lengths;

  std::ofstream out("a.txt");
  for (const auto & cell : cells) {
    double cell_length = (points[cell.vertices[1]] - points[cell.vertices[0]]).norm_square();
    out << cell.vertices[0] << " " << cell.vertices[1] << " " << cell_length << "\n";
  }
  out.close();

  int current_cell = 0;
  for (const auto & cell : cells) {
    if (adiacency[cell.vertices[1]].size() > 1 && adiacency[cell.vertices[0]].size() > 1 ) {
      lengths.push_back(resistances[current_cell]);
    }
    current_cell++;
  }

  std::sort(lengths.begin(), lengths.end(), [](const double & a, const double & b) {
    return a < b;
  });

  /*
  for (int i = 0; i < cells.size(); i++) {
    len << lengths[i] << "\n";
  }
  */

  unsigned int n_cells = cells.size();

  double median = lengths[(lengths.size() - 1) * coarsening_percentage] *1.00001;

  std::map<int, int> fine_to_coarse_vertex_map;

  //Vertici da visitare
  std::stack<int> to_visit_vertices;
  std::stack<int> to_visit_short_vertices;

  //Celle visitate
  std::vector<bool> visited_cell(n_cells, false);

  //Vettore che contiene true alla posizione i se l'i-esimo vertice
  //è da collassare (Non so come altro fare, forse c'è un modo migliore)
  std::vector<bool> is_collapsed(n_cells + 1, false);

  //I vertici da collassare, raggruppati per componente connessa
  std::vector<std::vector<int>> to_collapse_vertices;

  //Quando entro in short mode, inizio a riempire questo vector
  //Quando esco dalla short mode, lo pusho in to_collapse_vertices
  std::vector<int> current_to_collapse_vertices_connected_component;

  //Il primo vertice da visitare
  to_visit_vertices.push(0);

  bool short_mode = false;

  int visiting_vertex;

  while(to_visit_vertices.size() != 0 || to_visit_short_vertices.size() != 0) {

    if (short_mode) {
      visiting_vertex = to_visit_short_vertices.top();
      to_visit_short_vertices.pop();
    }
    else {
      visiting_vertex = to_visit_vertices.top();
      to_visit_vertices.pop();
    }
    
    //Se non sono in short mode, prima di esplorare controllo che almeno una cella sia corta,
    //Altrimenti rischio di aggiungere celle non corte che vengono visitate prima di quelle corte
    if (!short_mode) {

      for (unsigned int to_visit_cell : adiacency[visiting_vertex]) {
        if (!visited_cell[to_visit_cell]) {
          int other_vertex_cell = cells[to_visit_cell].vertices[1]+cells[to_visit_cell].vertices[0] - visiting_vertex;

          if (resistances[to_visit_cell] < median 
              && adiacency[other_vertex_cell].size() > 1 && adiacency[visiting_vertex].size() > 1
              && visiting_vertex != 0 ) {
            short_mode = true;
            break;
          }
        }
      }
    }

    //Metto il vertice in cui mi trovo adesso tra quelli da collassare
    if (short_mode) {
      current_to_collapse_vertices_connected_component.push_back(visiting_vertex);
      is_collapsed[visiting_vertex] = true;
    }

    for (unsigned int to_visit_cell : adiacency[visiting_vertex]) {
      int other_vertex_cell = cells[to_visit_cell].vertices[1]+cells[to_visit_cell].vertices[0] - visiting_vertex;

      if (!visited_cell[to_visit_cell]) {

        if (short_mode && resistances[to_visit_cell] < median 
            && adiacency[other_vertex_cell].size() > 1 && adiacency[visiting_vertex].size() > 1
            && visiting_vertex != 0) {
          to_visit_short_vertices.push(other_vertex_cell);
        }
        else {
          to_visit_vertices.push(other_vertex_cell);
        }

        visited_cell[to_visit_cell] = true;
      }
    }

    if (short_mode && to_visit_short_vertices.size() == 0) {
      to_collapse_vertices.push_back(current_to_collapse_vertices_connected_component);
      current_to_collapse_vertices_connected_component.clear();
      short_mode = false;
    }

  }

  //Creo i punti
  {
    for (unsigned int i = 0; i < n_cells + 1; i++) {
      if (!is_collapsed[i]){
        result.add_point(points[i]);

        fine_to_coarse_vertex_map[i] = result.get_number_of_points() - 1;

        coarse_to_fine_vertex_map[result.get_number_of_points() - 1].push_back(i);
      } 
    }

    int old_vertices = fine_to_coarse_vertex_map.size();

    for (unsigned int j = 0; j < to_collapse_vertices.size(); j++) {
      auto connected_componet = to_collapse_vertices[j];

      Point<3> new_vertex(0,0,0);

      for (unsigned int i = 0; i < connected_componet.size(); i++) {
        new_vertex += points[connected_componet[i]];

        fine_to_coarse_vertex_map[connected_componet[i]] = old_vertices+j;


        coarse_to_fine_vertex_map[old_vertices+j].push_back(connected_componet[i]);
      }

      //Il punto in mezzo
      new_vertex /= connected_componet.size();

      result.add_point(new_vertex);

      
    }
  }

  //Creo le celle
  {
    int current_fine_cell = 0;
    int current_coarse_cell = 0;

    for (const auto & cell : cells) 
    {

      if ( fine_to_coarse_vertex_map[cell.vertices[0]] != fine_to_coarse_vertex_map[cell.vertices[1]]) {
        result.add_cell(fine_to_coarse_vertex_map[cell.vertices[0]], fine_to_coarse_vertex_map[cell.vertices[1]]);

        coarse_to_fine_cell_map[current_coarse_cell] = current_fine_cell;
        current_coarse_cell++;

      }
      current_fine_cell++;

    }

  }

  return result;
}


Graph
Graph::get_coarser_graph_lort(std::map<int, int> & coarse_to_fine_vertex_map, 
  std::map<int, std::pair<std::pair<int, double>, std::pair<int, double>>> & not_trivial_prolongation) {

  Graph result;
  
  std::map<int, int> fine_to_coarse_vertex_map;

  bool no_remove = false;

  int i = 0;

  for(const auto & big_cell : big_cells) {
    int n_coarse_cell = 0;
    int coarse_cell_start = -1;
    int coarse_cell_end = -1;

    i++;
    //std::cout<<i<<"\n";
    
    if(big_cell.n_of_cells <= 5 && big_cell.node1 == big_cell.node2)
    {
      no_remove = true;
    }
    else
    {
      no_remove = false;
    }

    //Se il primo vertice non è presente, lo aggiungo
    if (fine_to_coarse_vertex_map.count(big_cell.node1) == 0) {
      result.add_point(points[big_cell.node1]);
      fine_to_coarse_vertex_map[big_cell.node1] = result.points.size()-1;

      coarse_to_fine_vertex_map[result.points.size()-1] = big_cell.node1;
    }

    int to_remove = 0;
    int last_hold_vertex = big_cell.node1;

    int current_vertex = big_cell.node1;
    int current_cell = big_cell.cell_start;

    //Finché non arrivo alla fine della big cell continuo a iterare
    do {
      //Cambio il vertice della cella
      current_vertex = cells[current_cell].vertices[1] + cells[current_cell].vertices[0] - current_vertex;

      if (current_vertex != big_cell.node2) {
        //Cambia la rimozione
        to_remove = 1-to_remove;

        //Se non è da rimuovere (per alternamento o perché la cella è da non coarseare)
        if (to_remove == 0 || no_remove) {  
          //Aggiungi il punto
          result.add_point(points[current_vertex]);
          fine_to_coarse_vertex_map[current_vertex] = result.points.size() - 1;

          coarse_to_fine_vertex_map[result.points.size()-1] = current_vertex;


          //Aggiungi il collegamento con il punto precedente
          result.add_cell(fine_to_coarse_vertex_map[last_hold_vertex], fine_to_coarse_vertex_map[current_vertex]);
          result.small_to_big_cell_map[result.cells.size()-1] = result.big_cells.size();

          if (coarse_cell_start < 0) {
            coarse_cell_start = result.cells.size()-1;
          }
          
          n_coarse_cell++;

          last_hold_vertex = current_vertex;
                                  if(last_hold_vertex == big_cell.node2) std::cout<<"AAA\n";

        }
        else {
          //SE PROVO A FARE INVERSE SI ROMPE QUI
          //b = 1/2 a + 1/2 c;
          std::pair<int, double> left_vertex_pair;
          int left_vertex = cells[adiacency[current_vertex][0]].vertices[0] + cells[adiacency[current_vertex][0]].vertices[1] - current_vertex;
          left_vertex_pair.first = fine_to_coarse_vertex_map[left_vertex]; 

          std::pair<int, double> right_vertex_pair;
          int right_vertex = cells[adiacency[current_vertex][1]].vertices[0] + cells[adiacency[current_vertex][1]].vertices[1] - current_vertex;
          if (right_vertex == big_cell.node2 && fine_to_coarse_vertex_map.count(big_cell.node2) != 0) 
            right_vertex_pair.first = fine_to_coarse_vertex_map[big_cell.node2];
          else
            right_vertex_pair.first = result.points.size(); //Il punto che devo ancora aggiungere

          double left_cell_length = (points[current_vertex] - points[left_vertex]).norm();
          double right_cell_length = (points[current_vertex] - points[right_vertex]).norm();

          left_vertex_pair.second = right_cell_length/(left_cell_length + right_cell_length);
          right_vertex_pair.second = left_cell_length/(left_cell_length + right_cell_length);

          not_trivial_prolongation[current_vertex] = std::make_pair(left_vertex_pair, right_vertex_pair);
        }

        //Cambia cella
        current_cell = adiacency[current_vertex][0] +  adiacency[current_vertex][1] - current_cell;
      }
    } while(current_vertex != big_cell.node2);

    //Se il secondo vertice non è presente, lo aggiungo
    if (fine_to_coarse_vertex_map.count(big_cell.node2) == 0) {
      result.add_point(points[big_cell.node2]);
      fine_to_coarse_vertex_map[big_cell.node2] = result.points.size()-1;

      coarse_to_fine_vertex_map[result.points.size()-1] = big_cell.node2;

    }

    //Devo creare il ponte tra l'ultimo vertice e il node2
    result.add_cell(fine_to_coarse_vertex_map[last_hold_vertex], fine_to_coarse_vertex_map[big_cell.node2]);
    result.small_to_big_cell_map[result.cells.size()-1] = result.big_cells.size();
    coarse_cell_end = result.cells.size()-1;
    n_coarse_cell++;

    if (coarse_cell_start < 0) {
      coarse_cell_start = result.cells.size()-1;
    }

    result.add_big_cell(fine_to_coarse_vertex_map[big_cell.node1], fine_to_coarse_vertex_map[big_cell.node2],
      coarse_cell_start, coarse_cell_end,
      n_coarse_cell);
  }

  //Le big cell sono le stesse
  result.dirichlet_big_cells = this -> dirichlet_big_cells;
  result.neumann_big_cells = this -> neumann_big_cells;

  return result;
}

Graph
Graph::get_finer_graph(int n_of_points_per_segment) {
  Graph finer_graph;

  for (const auto & point : points) {
    finer_graph.add_point(point);
  }

  for (const auto & cell : cells) {
    Point<3> a = points[cell.vertices[0]];
    Point<3> b = points[cell.vertices[1]];

    for (int i = 0; i < n_of_points_per_segment; i++) {
      //Per esempio, se n_of_points_per_segment = 2, ho che
      //Per i = 0, ho a * (1/3) + b * (2/3)
      //Per i = 1, ho a * (2/3) + b * (1/3)
      Point<3> new_point = (a * double(i+1) + b * double(n_of_points_per_segment - i)) /double(n_of_points_per_segment + 1);

      finer_graph.add_point(new_point);

      if (i == 0)
        finer_graph.add_cell(cell.vertices[0], finer_graph.get_number_of_points() - 1);
      else
        finer_graph.add_cell(finer_graph.get_number_of_points() - 2, finer_graph.get_number_of_points() - 1);

    }

    finer_graph.add_cell(finer_graph.get_number_of_points() - 1, cell.vertices[1]);
  }

  return finer_graph;
}